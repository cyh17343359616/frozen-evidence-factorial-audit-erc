"""
Qwen2.5-Omni 多模态特征提取器 v2
改进视频特征提取质量：
  1. vision-only pooling: 只对视觉token做mean pooling，排除prompt/system token
  2. last-token: 取序列最后一个token作为全局语义表示
  3. multi-layer: 多层隐藏状态加权聚合

背景：
  v1版本对整个序列做mean pooling，导致大量prompt文本token稀释了视觉信号。
  消融实验显示 video_only F1仅25.28%，视频特征几乎不携带情感信息。

Author: 陈裕瀚
"""
import os
import sys
import traceback
import torch
import numpy as np
import pandas as pd
import warnings
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

# MELD 情感映射
EMOTION_MAP = {
    'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3,
    'joy': 4, 'disgust': 5, 'anger': 6
}

# Qwen2.5-Omni 中视觉相关的特殊 token
# 视频帧被编码为 <|vision_bos|> ... <|vision_eos|> 之间的 <|VIDEO|> token
# 典型序列结构 (总长约1469):
#   [0-9]    system prompt
#   [10-13]  <|im_start|>user\n
#   [14]     <|vision_bos|>
#   [15-1454] 1440 × <|VIDEO|>  ← 视频token，占序列98%
#   [1455]   <|vision_eos|>
#   [1456-1463] prompt text "Describe the emotion..."
#   [1464-1468] <|im_end|>\n<|im_start|>assistant\n
VISION_START_TOKEN = '<|vision_bos|>'
VISION_END_TOKEN = '<|vision_eos|>'
VIDEO_PLACEHOLDER_TOKEN = '<|VIDEO|>'


class QwenFeatureExtractorV2:
    """
    改进版 Qwen2.5-Omni 特征提取器
    
    支持 4 种视频特征提取策略：
    - 'last_token': 取最后一个token的隐藏状态（推荐，模型的全局语义压缩）
    - 'vision_only': 只对<|VIDEO|>视觉token做pooling
    - 'multi_layer': 最后4层隐藏状态加权聚合 + last_token
    - 'mean_pool': v1兼容模式，对所有token做mean pooling
    
    注意：诊断发现 <|VIDEO|> token 占序列的 98%（1440/1469），
    因此 vision_only ≈ mean_pool。last_token 是唯一产生显著不同特征的策略。
    """
    
    def __init__(self, model_path: str, device: str = "auto",
                 video_pooling: str = "last_token",
                 text_pooling: str = "last_token",
                 num_layers_aggregate: int = 4,
                 torch_dtype: str = "auto"):
        """
        Args:
            model_path: Qwen2.5-Omni 模型路径
            device: 设备 ('auto', 'cuda', 'cpu')
            video_pooling: 视频特征pooling策略
            text_pooling: 文本特征pooling策略
            num_layers_aggregate: multi_layer模式使用最后几层
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
        self.hidden_size = None
        
        self.video_pooling = video_pooling
        self.text_pooling = text_pooling
        self.num_layers_aggregate = num_layers_aggregate
        self.torch_dtype = torch_dtype
        
        # 特殊token的ID（加载模型后初始化）
        self._vision_start_id = None
        self._vision_end_id = None
    
    def load_model(self):
        """加载 Qwen2.5-Omni 模型"""
        if self._loaded:
            return
        
        print(f"🚀 加载模型: {self.model_path}")
        print(f"   视频pooling策略: {self.video_pooling}")
        print(f"   文本pooling策略: {self.text_pooling}")
        
        try:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoProcessor
            ModelClass = Qwen2_5OmniThinkerForConditionalGeneration
            thinker_only = True
        except ImportError:
            from transformers import AutoModelForCausalLM as ModelClass, AutoProcessor
            thinker_only = False
            print("⚠ 使用 AutoModelForCausalLM 加载")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.torch_dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
                load_dtype = torch.float16
            else:
                load_dtype = torch.bfloat16
        else:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            if self.torch_dtype not in dtype_map:
                raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype}")
            load_dtype = dtype_map[self.torch_dtype]
        print(f"   模型加载 dtype: {load_dtype}")
        full_model = ModelClass.from_pretrained(
            self.model_path,
            torch_dtype=load_dtype,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()
        
        # Load the Thinker class directly.  In transformers 4.52.4 the full
        # Omni class unconditionally reads spk_dict.pt with torch.load after
        # loading weights, even when enable_audio_output=False; torch 2.5.1 is
        # intentionally preserved in the frozen Step 19 environment.
        if thinker_only:
            self.model = full_model
            print("✓ 直接加载 Thinker 子模型进行特征提取")
        elif hasattr(full_model, 'thinker'):
            self.model = full_model.thinker
            print("✓ 使用 model.thinker 子模型进行特征提取")
        else:
            self.model = full_model
            print("⚠ 未找到 thinker 子模型，使用完整模型")
        
        # 获取隐层维度
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
        else:
            self.hidden_size = 3584
        
        # 获取视觉特殊token的ID
        self._init_special_token_ids()
        
        self._loaded = True
        print(f"✓ 模型加载完成, hidden_size={self.hidden_size}")
    
    def _init_special_token_ids(self):
        """获取视觉相关特殊token的ID"""
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # 尝试获取 vision_start / vision_end token ID
        try:
            vocab = tokenizer.get_vocab()
            self._vision_start_id = vocab.get(VISION_START_TOKEN)
            self._vision_end_id = vocab.get(VISION_END_TOKEN)
            
            if self._vision_start_id is not None:
                print(f"✓ 视觉token ID: start={self._vision_start_id}, end={self._vision_end_id}")
            else:
                # 尝试通过 encode 获取
                try:
                    self._vision_start_id = tokenizer.encode(VISION_START_TOKEN, add_special_tokens=False)[0]
                    self._vision_end_id = tokenizer.encode(VISION_END_TOKEN, add_special_tokens=False)[0]
                    print(f"✓ 视觉token ID (encode): start={self._vision_start_id}, end={self._vision_end_id}")
                except:
                    print("⚠ 无法获取视觉token ID，vision_only模式将回退到mean_pool")
        except Exception as e:
            print(f"⚠ 获取特殊token失败: {e}")
    
    def _get_device(self):
        """安全获取模型所在设备"""
        if hasattr(self.model, 'device'):
            return self.model.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _forward_with_hidden_states(self, messages: List[Dict]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        前向传播并获取隐藏状态
        
        Returns:
            last_hidden: [1, seq_len, D] 最后一层隐藏状态
            all_hidden_states: list of [1, seq_len, D] 所有层隐藏状态
            input_ids: [1, seq_len] 输入token ID序列
        """
        device = self._get_device()
        
        try:
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Qwen2_5OmniProcessor in transformers 4.52.4 already returns a
            # list for a single conversation.  Wrapping that value again as
            # [text] creates a nested list, which later reaches re.finditer
            # and raises "expected string or bytes-like object".
            processor_text = text if isinstance(text, list) else [text]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=processor_text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
        except ImportError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            processor_text = text if isinstance(text, list) else [text]
            inputs = self.processor(
                text=processor_text, return_tensors="pt"
            ).to(device)
        
        # 保存 input_ids 用于定位 vision token
        input_ids = inputs.get('input_ids', None)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # 获取所有隐藏层状态
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            all_hidden = list(outputs.hidden_states)  # (num_layers+1) x [1, L, D]
            last_hidden = all_hidden[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
            all_hidden = [last_hidden]  # 回退
        else:
            raise RuntimeError("模型不支持 output_hidden_states")
        
        return last_hidden, all_hidden, input_ids
    
    def _find_vision_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        定位 vision token 的位置掩码
        
        在 Qwen2.5 系列中，视觉 token 位于 <|vision_start|> 和 <|vision_end|> 之间。
        
        Args:
            input_ids: [1, seq_len]
        
        Returns:
            mask: [1, seq_len] bool tensor, True 表示是 vision token
        """
        if input_ids is None or self._vision_start_id is None or self._vision_end_id is None:
            return None
        
        ids = input_ids[0]  # [seq_len]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        
        in_vision = False
        for i, token_id in enumerate(ids):
            tid = token_id.item()
            if tid == self._vision_start_id:
                in_vision = True
                continue  # 不包含 start token 本身
            elif tid == self._vision_end_id:
                in_vision = False
                continue  # 不包含 end token 本身
            
            if in_vision:
                mask[i] = True
        
        return mask.unsqueeze(0)  # [1, seq_len]
    
    def _pool_features(self, hidden: torch.Tensor, all_hidden: List[torch.Tensor],
                       input_ids: torch.Tensor, pooling: str,
                       is_video: bool = False) -> torch.Tensor:
        """
        根据策略池化特征
        
        Args:
            hidden: [1, L, D] 最后一层
            all_hidden: list of [1, L, D] 所有层
            input_ids: [1, L]
            pooling: pooling策略
            is_video: 是否为视频特征（影响vision_only策略）
        
        Returns:
            feature: [D]
        """
        if pooling == 'vision_only' and is_video:
            # 策略1: 只对 vision token 做 mean pooling
            vision_mask = self._find_vision_token_mask(input_ids)
            
            if vision_mask is not None and vision_mask.sum() > 0:
                # 只取 vision token 位置的隐藏状态
                vision_hidden = hidden[0][vision_mask[0]]  # [num_vision_tokens, D]
                feature = vision_hidden.mean(dim=0)  # [D]
                return feature
            else:
                # 回退到 mean_pool
                print("  ⚠ 未找到视觉token，回退到mean_pool")
                feature = hidden.mean(dim=1).squeeze(0)
                return feature
        
        elif pooling == 'last_token':
            # 策略2: 取最后一个token
            feature = hidden[0, -1, :]  # [D]
            return feature
        
        elif pooling == 'multi_layer':
            # 策略3: 最后 N 层加权聚合
            n = min(self.num_layers_aggregate, len(all_hidden))
            layers = all_hidden[-n:]  # 最后 n 层
            
            # 可学习权重（这里用简单的线性衰减：越靠近最后一层权重越大）
            weights = torch.arange(1, n + 1, dtype=torch.float32, device=hidden.device)
            weights = weights / weights.sum()  # 归一化
            
            # 加权聚合
            aggregated = torch.zeros_like(layers[0])
            for w, layer_hidden in zip(weights, layers):
                aggregated += w * layer_hidden
            
            feature = aggregated.mean(dim=1).squeeze(0)  # [D]
            return feature
        
        else:  # 'mean_pool' 或默认
            # v1 兼容模式
            feature = hidden.mean(dim=1).squeeze(0)
            return feature
    
    def extract_video_feature(self, video_path: str) -> np.ndarray:
        """
        提取视频特征向量（改进版）
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            feature: [D] 特征向量
        """
        if not self._loaded:
            self.load_model()
        
        # 视频消息：仅包含视频，不加指令文本
        # 原v1用 "Describe the emotion shown in this video." 但指令token
        # 占比很小（8/1469），且可能引导模型偏向生成描述而非情感特征
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_frames": 16},
                    {"type": "text", "text": "What emotion is expressed?"}
                ]
            }
        ]
        
        torch.cuda.empty_cache()
        last_hidden, all_hidden, input_ids = self._forward_with_hidden_states(messages)
        
        feature = self._pool_features(
            last_hidden, all_hidden, input_ids,
            pooling=self.video_pooling, is_video=True
        )
        
        return feature.float().cpu().numpy()
    
    def extract_text_feature(self, text: str) -> np.ndarray:
        """
        提取纯文本特征向量
        
        Args:
            text: 输入文本（对白）
            
        Returns:
            feature: [D] 特征向量
        """
        if not self._loaded:
            self.load_model()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'What emotion does this utterance express: "{text}"'}
                ]
            }
        ]
        
        last_hidden, all_hidden, input_ids = self._forward_with_hidden_states(messages)
        
        feature = self._pool_features(
            last_hidden, all_hidden, input_ids,
            pooling=self.text_pooling, is_video=False
        )
        
        return feature.float().cpu().numpy()
    
    def extract_batch(self, samples: List[Dict],
                      save_dir: Optional[str] = None,
                      split_name: str = 'train',
                      suffix: str = '') -> Dict[str, np.ndarray]:
        """
        批量提取特征并缓存到磁盘
        
        Args:
            samples: 样本列表
            save_dir: 缓存目录
            split_name: 数据集划分名
            suffix: 文件名后缀（用于区分不同策略的特征，如 '_v2'）
        
        Returns:
            dict with 'video_features', 'text_features', 'labels'
        """
        if not self._loaded:
            self.load_model()
        
        video_features = []
        text_features = []
        labels = []
        dialogue_ids = []
        utterance_ids = []
        
        failed = 0
        vision_token_counts = []  # 记录每个样本的vision token数，用于诊断
        
        for i, sample in enumerate(tqdm(samples, desc=f"Extracting {split_name}")):
            try:
                video_feat = self.extract_video_feature(sample['video_path'])
                text_feat = self.extract_text_feature(sample['utterance'])
                
                video_features.append(video_feat)
                text_features.append(text_feat)
                labels.append(sample['true_label'])
                dialogue_ids.append(sample.get('dialogue_id', -1))
                utterance_ids.append(sample.get('utterance_id', -1))
                
                # 定期清理GPU缓存 + 进度报告
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if failed == 0:
                    print(f"\n⚠ 样本 {i} 详细错误:")
                    traceback.print_exc()
                else:
                    print(f"⚠ 跳过样本 {i}: {e}")
                failed += 1
                torch.cuda.empty_cache()
                continue
        
        result = {
            'video_features': np.stack(video_features),
            'text_features': np.stack(text_features),
            'labels': np.array(labels),
            'dialogue_ids': np.array(dialogue_ids),
            'utterance_ids': np.array(utterance_ids),
        }
        
        print(f"✓ 提取完成: {len(video_features)}/{len(samples)} 样本, {failed} 失败")
        print(f"  video_features: {result['video_features'].shape}")
        print(f"  text_features:  {result['text_features'].shape}")
        
        # 打印特征质量摘要
        v_feat = result['video_features']
        t_feat = result['text_features']
        print(f"\n📊 特征质量摘要:")
        print(f"  video: mean={v_feat.mean():.4f}, std={v_feat.std():.4f}, "
              f"L2_norm={np.linalg.norm(v_feat, axis=1).mean():.2f}")
        print(f"  text:  mean={t_feat.mean():.4f}, std={t_feat.std():.4f}, "
              f"L2_norm={np.linalg.norm(t_feat, axis=1).mean():.2f}")
        
        # 保存到磁盘
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for key, arr in result.items():
                path = os.path.join(save_dir, f"{split_name}_{key}{suffix}.npy")
                np.save(path, arr)
                print(f"  💾 保存: {path} {arr.shape}")
        
        return result


def load_cached_features(feature_dir: str, split: str) -> Dict[str, np.ndarray]:
    """
    加载缓存的特征文件
    
    支持 float16 和 float32 格式，自动转换为 float32。
    
    Args:
        feature_dir: 特征缓存目录
        split: 数据集划分 (train/dev/test)
        
    Returns:
        dict with numpy arrays
    """
    result = {}
    for key in [
        'video_features', 'text_features', 'labels', 'dialogue_ids', 'utterance_ids',
        'is_augmented', 'source_indices',
    ]:
        path = os.path.join(feature_dir, f"{split}_{key}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            # float16 特征自动转为 float32（训练需要 float32 精度）
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)
            result[key] = arr
            print(f"  📁 加载: {path} {result[key].shape} ({result[key].dtype})")
        else:
            if key in ('dialogue_ids', 'utterance_ids', 'is_augmented', 'source_indices'):
                continue  # 这两个是可选的
            raise FileNotFoundError(f"特征文件不存在: {path}")
    
    return result


def load_meld_samples(csv_path: str, video_dir: str, max_samples: int = None):
    """加载MELD样本列表"""
    df = pd.read_csv(csv_path)
    samples = []
    missing = 0
    
    for _, row in df.iterrows():
        if max_samples and len(samples) >= max_samples:
            break
            
        dialogue_id = int(row['Dialogue_ID'])
        utterance_id = int(row['Utterance_ID'])
        video_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_dir, video_name)
        
        if not os.path.exists(video_path):
            missing += 1
            continue
        
        if os.path.getsize(video_path) < 50000:
            missing += 1
            continue
        
        emotion = row['Emotion'].strip().lower()
        if emotion not in EMOTION_MAP:
            continue
        
        samples.append({
            'video_path': video_path,
            'utterance': str(row['Utterance']),
            'dialogue_id': dialogue_id,
            'utterance_id': utterance_id,
            'true_label': EMOTION_MAP[emotion],
            'true_emotion': emotion,
        })
    
    print(f"✓ 加载 {len(samples)} 个样本 (跳过 {missing} 个)")
    return samples


def find_video_dir(data_dir: Path, split: str) -> Path:
    """查找视频目录"""
    candidates = [
        data_dir / f'{split}_splits_complete',
        data_dir / f'{split}_splits',
        data_dir / split / f'output_repeated_splits_{split}',
    ]
    for d in candidates:
        if d.exists() and list(d.glob('dia*.mp4')):
            return d
    raise FileNotFoundError(f"找不到视频目录: {candidates}")


# ===================== 各策略对比提取脚本 =====================

def extract_with_comparison(model_path: str, data_dir: str, output_dir: str,
                            split: str = 'dev', max_samples: int = 50):
    """
    用多种策略提取小批量特征进行对比，用于快速验证哪种策略更优
    
    在 AutoDL 上运行：
        python src/feature_extractor_v2.py \
            --model_path /root/autodl-tmp/model/Qwen/Qwen2.5-Omni-7B \
            --data_dir /root/autodl-tmp/MELD/MELD.Raw \
            --compare --split dev --max_samples 50
    """
    # load_meld_samples 和 find_video_dir 已在本文件中定义
    
    csv_path = os.path.join(data_dir, f'{split}_sent_emo.csv')
    video_dir = find_video_dir(Path(data_dir), split)
    samples = load_meld_samples(csv_path, str(video_dir), max_samples)
    
    strategies = [
        ('mean_pool', 'mean_pool'),      # v1 基线
        ('last_token', 'last_token'),    # 推荐方案：last token
        ('multi_layer', 'last_token'),   # 多层聚合 + last token
    ]
    
    print("=" * 60)
    print("视频特征提取策略对比实验")
    print(f"样本数: {len(samples)}, split: {split}")
    print("=" * 60)
    
    for video_pool, text_pool in strategies:
        tag = f"{video_pool}__{text_pool}"
        print(f"\n{'─' * 60}")
        print(f"📌 策略: video={video_pool}, text={text_pool}")
        print(f"{'─' * 60}")
        
        extractor = QwenFeatureExtractorV2(
            model_path, video_pooling=video_pool, text_pooling=text_pool
        )
        extractor.load_model()
        
        result = extractor.extract_batch(
            samples, save_dir=output_dir,
            split_name=split, suffix=f'_{tag}'
        )
        
        # 释放模型显存
        del extractor
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("✓ 所有策略提取完成！")
    print(f"  结果保存在: {output_dir}")
    print("  请使用 diagnose_features.py 对比各策略特征质量")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Qwen2.5-Omni 特征提取 v2')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Qwen2.5-Omni 模型路径')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='MELD 数据集目录')
    parser.add_argument('--output_dir', type=str, default='datasets/MELD/features_v2',
                        help='特征输出目录')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                        help='要处理的数据集划分')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='每个划分最多处理的样本数')
    parser.add_argument('--video_pooling', type=str, default='last_token',
                        choices=['vision_only', 'last_token', 'multi_layer', 'mean_pool'],
                        help='视频特征pooling策略')
    parser.add_argument('--text_pooling', type=str, default='last_token',
                        choices=['last_token', 'mean_pool', 'multi_layer'],
                        help='文本特征pooling策略')
    parser.add_argument('--compare', action='store_true',
                        help='运行多策略对比模式（小样本快速验证）')
    parser.add_argument('--split', type=str, default='dev',
                        help='对比模式使用的split')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # 处理相对路径
    if not os.path.isabs(args.output_dir):
        project_root = Path(__file__).parent.parent.parent
        args.output_dir = str(project_root / args.output_dir)
    
    if args.compare:
        # 多策略对比模式
        extract_with_comparison(
            args.model_path, args.data_dir, args.output_dir,
            split=args.split, max_samples=args.max_samples or 50
        )
    else:
        # 正常提取模式
        # load_meld_samples 和 find_video_dir 已在本文件中定义
        
        data_dir = Path(args.data_dir)
        
        print("=" * 60)
        print("MELD 多模态特征提取 v2")
        print("=" * 60)
        print(f"模型路径: {args.model_path}")
        print(f"视频pooling: {args.video_pooling}")
        print(f"文本pooling: {args.text_pooling}")
        print(f"输出目录: {args.output_dir}")
        
        extractor = QwenFeatureExtractorV2(
            args.model_path, device=args.device,
            video_pooling=args.video_pooling,
            text_pooling=args.text_pooling
        )
        extractor.load_model()
        
        for i, split in enumerate(args.splits):
            print(f"\n[{i+1}] 处理 {split} 集...")
            
            csv_path = data_dir / f'{split}_sent_emo.csv'
            if not csv_path.exists():
                print(f"  ⚠ CSV文件不存在: {csv_path}")
                continue
            
            try:
                video_dir = find_video_dir(data_dir, split)
            except FileNotFoundError as e:
                print(f"  ⚠ {e}")
                continue
            
            samples = load_meld_samples(str(csv_path), str(video_dir), args.max_samples)
            
            if not samples:
                print(f"  ⚠ 没有有效样本")
                continue
            
            extractor.extract_batch(samples, save_dir=args.output_dir, split_name=split)
        
        print("\n" + "=" * 60)
        print("✓ 特征提取完成!")
        print("=" * 60)
