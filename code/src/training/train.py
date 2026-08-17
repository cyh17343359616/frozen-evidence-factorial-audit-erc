"""
多模态情感分类 + 风格预测 联合训练脚本

支持:
- 5种模式(text_only/video_only/concat/attention/gated)用于消融实验
- Focal Loss处理类别不平衡
- 早停 (early stopping)
- 模型保存与加载
- TensorBoard日志(可选)

用法:
    # 跨模态注意力融合训练
    python src/train.py --feature_dir datasets/MELD/features \
        --fusion attention --epochs 50 --lr 1e-3 --batch_size 64

    # 冒烟测试（用随机假数据）
    python src/train.py --smoke_test --fusion attention --epochs 3

Author: 陈裕瀚
"""
import os
import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent))

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import EmotionClassifier, FocalLoss, get_class_weights
from models.style_predictor import StylePredictor, EmotionStyleModel, generate_style_labels
from models.emotion_rag import EmotionMemoryBank, EmotionRAGClassifier
from data.feature_extractor_v2 import load_cached_features

# ===================== 数据集配置 =====================
DATASET_LABELS = {
    'meld': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
    'chsims': ['negative', 'neutral', 'positive'],
    'mosei': ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
}


RELATION_LABELS = {
    0: 'unknown',
    1: 'reply_to',
    2: 'question_answer',
    3: 'interruption',
}


def load_validated_relation_ids(path: str, split: str, expected_len: int) -> np.ndarray:
    relation_ids = np.zeros(expected_len, dtype=np.int64)
    relation_path = Path(path)
    if not relation_path.is_absolute():
        relation_path = Path(__file__).parent.parent.parent / relation_path
    if not relation_path.exists():
        raise FileNotFoundError(f"validated relations file not found: {relation_path}")

    loaded = 0
    with relation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            idx = int(row["sample_index"])
            if 0 <= idx < expected_len:
                relation_ids[idx] = int(row.get("primary_relation_id", 0))
                loaded += 1

    if loaded != expected_len:
        print(f"⚠ validated relations loaded {loaded}/{expected_len} for split={split}")
    return relation_ids


def build_speaker_prototypes(
    video_features: np.ndarray,
    text_features: np.ndarray,
    speaker_ids: Optional[np.ndarray],
    slots: int = 4,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build fixed unsupervised speaker prototype slots from the training split."""
    feat_dim = int(video_features.shape[1])
    prototypes: Dict[str, Dict[str, np.ndarray]] = {}
    empty_video = np.zeros((slots, feat_dim), dtype=np.float32)
    empty_text = np.zeros((slots, feat_dim), dtype=np.float32)
    empty_mask = np.zeros(slots, dtype=np.float32)

    if speaker_ids is None or len(speaker_ids) == 0:
        prototypes["__GLOBAL__"] = {"video": empty_video, "text": empty_text, "mask": empty_mask}
        return prototypes

    speakers = np.asarray([str(s) for s in speaker_ids])
    concat = np.concatenate([video_features, text_features], axis=1).astype(np.float32)

    for speaker in sorted(set(speakers.tolist())):
        indices = np.where(speakers == speaker)[0]
        speaker_video = np.zeros((slots, feat_dim), dtype=np.float32)
        speaker_text = np.zeros((slots, feat_dim), dtype=np.float32)
        mask = np.zeros(slots, dtype=np.float32)
        if len(indices) == 0:
            prototypes[speaker] = {"video": speaker_video, "text": speaker_text, "mask": mask}
            continue

        n_clusters = min(slots, len(indices))
        if n_clusters == 1:
            labels = np.zeros(len(indices), dtype=np.int64)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                batch_size=max(16, min(256, len(indices))),
                n_init=3,
            )
            labels = kmeans.fit_predict(concat[indices])

        cluster_rows = []
        for cluster_id in range(n_clusters):
            members = indices[labels == cluster_id]
            if len(members):
                cluster_rows.append((len(members), cluster_id, members))
        cluster_rows.sort(reverse=True, key=lambda item: item[0])

        for slot_idx, (_, _, members) in enumerate(cluster_rows[:slots]):
            speaker_video[slot_idx] = video_features[members].mean(axis=0)
            speaker_text[slot_idx] = text_features[members].mean(axis=0)
            mask[slot_idx] = 1.0
        prototypes[speaker] = {"video": speaker_video, "text": speaker_text, "mask": mask}

    global_video = np.zeros((slots, feat_dim), dtype=np.float32)
    global_text = np.zeros((slots, feat_dim), dtype=np.float32)
    global_mask = np.zeros(slots, dtype=np.float32)
    if len(video_features):
        global_video[0] = video_features.mean(axis=0)
        global_text[0] = text_features.mean(axis=0)
        global_mask[0] = 1.0
    prototypes["__GLOBAL__"] = {"video": global_video, "text": global_text, "mask": global_mask}
    return prototypes


def save_speaker_prototypes(path: str | Path, prototypes: Dict[str, Dict[str, np.ndarray]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        speakers=np.array(list(prototypes.keys()), dtype=object),
        video=np.stack([item["video"] for item in prototypes.values()]),
        text=np.stack([item["text"] for item in prototypes.values()]),
        mask=np.stack([item["mask"] for item in prototypes.values()]),
    )


def load_speaker_prototypes(path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    prototypes = {}
    for speaker, video, text, mask in zip(data["speakers"], data["video"], data["text"], data["mask"]):
        prototypes[str(speaker)] = {
            "video": video.astype(np.float32),
            "text": text.astype(np.float32),
            "mask": mask.astype(np.float32),
        }
    return prototypes


# ===================== 数据集 =====================

class MELDFeatureDataset(Dataset):
    """加载缓存的特征文件的Dataset"""
    
    def __init__(self, video_features: np.ndarray, text_features: np.ndarray,
                 labels: np.ndarray, dialogue_ids: Optional[np.ndarray] = None,
                 utterance_ids: Optional[np.ndarray] = None,
                 speaker_ids: Optional[np.ndarray] = None,
                 relation_ids: Optional[np.ndarray] = None,
                 is_augmented: Optional[np.ndarray] = None,
                 source_indices: Optional[np.ndarray] = None,
                 context_len: int = 5,
                 memory_bank: Optional[EmotionMemoryBank] = None,
                 rag_mode: str = 'none', rag_top_k: int = 5,
                 context_max_distance: int = 20,
                 speaker_prototypes: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 speaker_memory_slots: int = 4):
        """
        Args:
            video_features: [N, D_v]
            text_features: [N, D_t]
            labels: [N]
            dialogue_ids: [N] 对话ID（用于构建上下文）
            utterance_ids: [N] 说话轮次ID
            speaker_ids: [N] 说话人ID（用于 RAG-Speaker）
            relation_ids: [N] 当前轮相对上一轮的 validated relation id
            context_len: 上下文窗口大小
            memory_bank: EmotionMemoryBank 实例（RAG 模式用）
            rag_mode: 'none' / 'global' / 'speaker' / 'dialogue'
            rag_top_k: RAG 检索数量
            context_max_distance: turn distance embedding 的最大距离
            speaker_prototypes: 固定 offline speaker prototype slots
            speaker_memory_slots: 每个 speaker 的 prototype slot 数
        """
        # 过滤掉由于视频损坏而提取为全 0 的特征
        video_norms = np.linalg.norm(video_features, axis=1)
        valid_indices = np.where(video_norms > 1e-6)[0]
        
        if len(valid_indices) < len(labels):
            print(f"  ⚡ 自动过滤 {len(labels) - len(valid_indices)} 个无效视频特征 (全0向量)")
            video_features = video_features[valid_indices]
            text_features = text_features[valid_indices]
            labels = labels[valid_indices]
            if dialogue_ids is not None:
                dialogue_ids = dialogue_ids[valid_indices]
            if utterance_ids is not None:
                utterance_ids = utterance_ids[valid_indices]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[valid_indices]
            if relation_ids is not None:
                relation_ids = relation_ids[valid_indices]
            if is_augmented is not None:
                is_augmented = is_augmented[valid_indices]
            if source_indices is not None:
                source_indices = source_indices[valid_indices]
        
        # L2 归一化：原始特征范数可能很大，需要归一化
        video_t = torch.tensor(video_features, dtype=torch.float32)
        text_t = torch.tensor(text_features, dtype=torch.float32)
        self.video_features = torch.nn.functional.normalize(video_t, p=2, dim=1)
        self.text_features = torch.nn.functional.normalize(text_t, p=2, dim=1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.context_len = context_len
        self.context_max_distance = context_max_distance
        
        # 构建对话索引（用于上下文）
        self.dialogue_ids = dialogue_ids
        self.utterance_ids = utterance_ids
        self.speaker_ids = speaker_ids
        self.relation_ids = relation_ids
        self.is_augmented = (
            np.asarray(is_augmented, dtype=bool)
            if is_augmented is not None else np.zeros(len(labels), dtype=bool)
        )
        self.source_indices = (
            np.asarray(source_indices, dtype=np.int64)
            if source_indices is not None else np.arange(len(labels), dtype=np.int64)
        )
        self.speaker_prototypes = speaker_prototypes
        self.speaker_memory_slots = speaker_memory_slots
        self._build_dialogue_index()
        
        # RAG 配置
        self.memory_bank = memory_bank
        self.rag_mode = rag_mode
        self.rag_top_k = rag_top_k
        
        # 预计算的 RAG 融合特征缓存（训练前由外部填充）
        self._rag_cache = None  # {idx: (features, labels, scores)}
    
    def _build_dialogue_index(self):
        """构建 dialogue_id -> [样本索引列表] 的映射"""
        self.dialogue_map = {}
        self.position_map = {}
        if self.dialogue_ids is not None:
            for idx, (did, uid) in enumerate(zip(self.dialogue_ids, self.utterance_ids)):
                if self.is_augmented[idx]:
                    continue
                did = str(did)
                # 兼容不能直接转换为 int 的 uid 字符串 (尝试转为 float 排序)
                try:
                    num_uid = float(uid)
                except ValueError:
                    num_uid = str(uid)
                    
                if did not in self.dialogue_map:
                    self.dialogue_map[did] = []
                self.dialogue_map[did].append((num_uid, idx))
                
            # 按 utterance_id (num_uid) 排序
            for did in self.dialogue_map:
                self.dialogue_map[did].sort(key=lambda x: x[0])
                for position, (_, sample_idx) in enumerate(self.dialogue_map[did]):
                    self.position_map[sample_idx] = position
            for idx in np.where(self.is_augmented)[0]:
                source_idx = int(self.source_indices[idx])
                if source_idx not in self.position_map:
                    raise ValueError(f"Augmented row {idx} has invalid source index {source_idx}")
                self.position_map[idx] = self.position_map[source_idx]
    
    def _same_speaker(self, left_idx: int, right_idx: int) -> float:
        if self.speaker_ids is None:
            return 0.0
        left = self.speaker_ids[left_idx]
        right = self.speaker_ids[right_idx]
        return 1.0 if str(left) == str(right) else 0.0

    def _get_context(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取样本idx的对话上下文（前k轮特征 + 稳定结构字段）
        
        Returns:
            context_labels: [context_len, 7] one-hot编码的情感序列
            context_mask: [context_len] 有效位掩码
            context_video: [context_len, video_dim]
            context_text: [context_len, text_dim]
            context_same_speaker: [context_len] 当前轮与上下文轮是否同一说话人
            context_turn_distance: [context_len] 当前轮与上下文轮的 turn distance
            context_relation_ids: [context_len] 当前轮与上下文轮的 validated relation id；非相邻轮默认为 unknown
        """
        context_labels = torch.zeros(self.context_len, 7)
        context_mask = torch.zeros(self.context_len)
        context_video = torch.zeros(self.context_len, self.video_features.size(1), dtype=torch.float32)
        context_text = torch.zeros(self.context_len, self.text_features.size(1), dtype=torch.float32)
        context_same_speaker = torch.zeros(self.context_len, dtype=torch.float32)
        context_turn_distance = torch.zeros(self.context_len, dtype=torch.long)
        context_relation_ids = torch.zeros(self.context_len, dtype=torch.long)
        
        if self.dialogue_ids is None or len(self.dialogue_map) == 0:
            return context_labels, context_mask, context_video, context_text, context_same_speaker, context_turn_distance, context_relation_ids
        
        did = str(self.dialogue_ids[idx])
        uid = self.utterance_ids[idx]
        try:
            num_uid = float(uid)
        except ValueError:
            num_uid = str(uid)
        
        if did not in self.dialogue_map:
            return context_labels, context_mask, context_video, context_text, context_same_speaker, context_turn_distance, context_relation_ids
        
        # 找到当前utterance之前的轮次
        dialogue = self.dialogue_map[did]
        current_position = self.position_map.get(idx)
        if current_position is not None:
            prev_items = [(pos, sample_idx) for pos, (_, sample_idx) in enumerate(dialogue) if pos < current_position]
        else:
            prev_items = [(None, sample_idx) for u_id, sample_idx in dialogue if u_id < num_uid]
        
        # 取最近 context_len 轮
        prev_items = prev_items[-self.context_len:]
        
        for i, (prev_position, prev_idx) in enumerate(prev_items):
            label = self.labels[prev_idx]
            offset = self.context_len - len(prev_items) + i
            context_labels[offset, label] = 1.0
            context_mask[offset] = 1.0
            context_video[offset] = self.video_features[prev_idx]
            context_text[offset] = self.text_features[prev_idx]
            context_same_speaker[offset] = self._same_speaker(idx, prev_idx)

            if current_position is not None and prev_position is not None:
                distance = current_position - prev_position
            else:
                try:
                    distance = int(float(num_uid) - float(self.utterance_ids[prev_idx]))
                except (TypeError, ValueError):
                    distance = len(prev_items) - i
            distance = max(1, min(int(distance), self.context_max_distance))
            context_turn_distance[offset] = distance
            if self.relation_ids is not None and current_position is not None and prev_position == current_position - 1:
                context_relation_ids[offset] = int(self.relation_ids[idx])
        
        return context_labels, context_mask, context_video, context_text, context_same_speaker, context_turn_distance, context_relation_ids
    
    def _prepare_context_labels(self):
        # Infer num_classes from labels max + 1
        num_classes = int(self.labels.max().item() + 1)
        self.num_classes = num_classes

    def _get_speaker_memory(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.speaker_memory_slots
        feat_dim = self.video_features.size(1)
        empty_video = torch.zeros(slots, feat_dim, dtype=torch.float32)
        empty_text = torch.zeros(slots, feat_dim, dtype=torch.float32)
        empty_mask = torch.zeros(slots, dtype=torch.float32)

        if not self.speaker_prototypes or self.speaker_ids is None:
            return empty_video, empty_text, empty_mask

        speaker = str(self.speaker_ids[idx])
        proto = self.speaker_prototypes.get(speaker) or self.speaker_prototypes.get("__GLOBAL__")
        if proto is None:
            return empty_video, empty_text, empty_mask

        video = torch.tensor(proto["video"], dtype=torch.float32)
        text = torch.tensor(proto["text"], dtype=torch.float32)
        mask = torch.tensor(proto["mask"], dtype=torch.float32)
        if video.size(0) == slots:
            return video, text, mask

        fixed_video = empty_video.clone()
        fixed_text = empty_text.clone()
        fixed_mask = empty_mask.clone()
        keep = min(slots, video.size(0))
        fixed_video[:keep] = video[:keep]
        fixed_text[:keep] = text[:keep]
        fixed_mask[:keep] = mask[:keep]
        return fixed_video, fixed_text, fixed_mask
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        (
            context_labels,
            context_mask,
            context_video,
            context_text,
            context_same_speaker,
            context_turn_distance,
            context_relation_ids,
        ) = self._get_context(idx)
        memory_video, memory_text, memory_mask = self._get_speaker_memory(idx)
        
        item = {
            'video_feat': self.video_features[idx],
            'text_feat': self.text_features[idx],
            'label': self.labels[idx],
            'context_emotions': context_labels,
            'context_mask': context_mask,
            'context_video_feat': context_video,
            'context_text_feat': context_text,
            'context_same_speaker': context_same_speaker,
            'context_turn_distance': context_turn_distance,
            'context_relation_ids': context_relation_ids,
            'speaker_memory_video_feat': memory_video,
            'speaker_memory_text_feat': memory_text,
            'speaker_memory_mask': memory_mask,
            'idx': idx,
        }
        
        # RAG 检索结果（从预计算缓存中读取）
        if self._rag_cache is not None and idx in self._rag_cache:
            r_feat, r_labels, r_scores = self._rag_cache[idx]
            item['retrieved_features'] = torch.tensor(r_feat, dtype=torch.float32)
            item['retrieved_labels'] = torch.tensor(r_labels, dtype=torch.long)
            item['retrieved_scores'] = torch.tensor(r_scores, dtype=torch.float32)
        
        return item


def create_smoke_test_data(num_samples: int = 200, feat_dim: int = 3584):
    """生成冒烟测试用的随机假数据"""
    print("🔧 生成冒烟测试假数据...")
    data = {
        'video_features': np.random.randn(num_samples, feat_dim).astype(np.float32),
        'text_features': np.random.randn(num_samples, feat_dim).astype(np.float32),
        'labels': np.random.randint(0, 7, num_samples),
        'dialogue_ids': np.repeat(np.arange(num_samples // 5), 5)[:num_samples],
        'utterance_ids': np.tile(np.arange(5), num_samples // 5)[:num_samples],
        'speakers': np.tile(np.array(["A", "B", "C", "D", "E"], dtype=object), num_samples // 5)[:num_samples],
    }
    return data


# ===================== 训练器 =====================

class Trainer:
    """训练管理器"""
    
    def __init__(self, model: nn.Module, args, device: torch.device):
        self.model = model.to(device)
        self.args = args
        self.device = device
        
        # 损失函数。保留 --use_focal_loss 作为旧命令的兼容别名。
        loss_type = getattr(args, 'loss_type', 'ce')
        legacy_focal = getattr(args, 'use_focal_loss', False)
        if legacy_focal:
            loss_type = 'focal'
        weight_mode = getattr(args, 'class_weight_mode', 'none')
        if legacy_focal and weight_mode == 'none':
            weight_mode = 'inverse'
        class_weights = None
        if weight_mode != 'none':
            class_weights = get_class_weights(
                args.dataset,
                getattr(args, 'label_counts', None),
                mode=weight_mode,
            ).to(device)
        if loss_type == 'focal':
            self.criterion_emotion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        else:
            self.criterion_emotion = nn.CrossEntropyLoss(weight=class_weights)
        print(
            f"📌 Emotion loss: type={loss_type}, class_weight_mode={weight_mode}, "
            f"focal_gamma={args.focal_gamma if loss_type == 'focal' else '-'}, "
            f"weights={class_weights.detach().cpu().tolist() if class_weights is not None else None}"
        )
        self.criterion_style = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        
        # 早停
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.best_epoch = 0
        
        # 日志
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_macro_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_macro_f1': [],
        }

    def _context_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            'context_video_feat': batch['context_video_feat'].to(self.device),
            'context_text_feat': batch['context_text_feat'].to(self.device),
            'context_mask': batch['context_mask'].to(self.device),
            'context_same_speaker': batch['context_same_speaker'].to(self.device),
            'context_turn_distance': batch['context_turn_distance'].to(self.device),
            'context_relation_ids': batch['context_relation_ids'].to(self.device),
            'speaker_memory_video_feat': batch['speaker_memory_video_feat'].to(self.device),
            'speaker_memory_text_feat': batch['speaker_memory_text_feat'].to(self.device),
            'speaker_memory_mask': batch['speaker_memory_mask'].to(self.device),
        }

    def _accumulate_attention_stats(self, output: Dict[str, Any], stats: Dict[str, float]) -> None:
        weights = output.get('attention_weights') or {}
        if 'gate' in weights and weights['gate'] is not None:
            gate = weights['gate'].detach()
            stats['fusion_gate_sum'] += float(gate.sum().item())
            stats['fusion_gate_count'] += int(gate.numel())
        if 'projection_cosine' in weights and weights['projection_cosine'] is not None:
            cosine = weights['projection_cosine'].detach()
            stats['projection_cosine_sum'] += float(cosine.sum().item())
            stats['projection_cosine_count'] += int(cosine.numel())
        if 'projection_text_norm' in weights and weights['projection_text_norm'] is not None:
            norm = weights['projection_text_norm'].detach()
            stats['projection_text_norm_sum'] += float(norm.sum().item())
            stats['projection_text_norm_count'] += int(norm.numel())
        if 'projection_video_norm' in weights and weights['projection_video_norm'] is not None:
            norm = weights['projection_video_norm'].detach()
            stats['projection_video_norm_sum'] += float(norm.sum().item())
            stats['projection_video_norm_count'] += int(norm.numel())
        if 'disagreement_gate' in weights and weights['disagreement_gate'] is not None:
            gate = weights['disagreement_gate'].detach()
            stats['disagreement_gate_sum'] += float(gate.sum().item())
            stats['disagreement_gate_count'] += int(gate.numel())
        if 'js_divergence' in weights and weights['js_divergence'] is not None:
            js = weights['js_divergence'].detach()
            stats['js_divergence_sum'] += float(js.sum().item())
            stats['js_divergence_count'] += int(js.numel())
        if 'kl_text_video' in weights and weights['kl_text_video'] is not None:
            kl = weights['kl_text_video'].detach()
            stats['kl_text_video_sum'] += float(kl.sum().item())
            stats['kl_text_video_count'] += int(kl.numel())
        if 'kl_video_text' in weights and weights['kl_video_text'] is not None:
            kl = weights['kl_video_text'].detach()
            stats['kl_video_text_sum'] += float(kl.sum().item())
            stats['kl_video_text_count'] += int(kl.numel())
        if 'structure_gate' in weights and weights['structure_gate'] is not None:
            gate = weights['structure_gate'].detach()
            stats['structure_gate_sum'] += float(gate.sum().item())
            stats['structure_gate_count'] += int(gate.numel())
        if 'structure_attention' in weights and weights['structure_attention'] is not None:
            attn = weights['structure_attention'].detach()
            if attn.numel() > 0:
                max_attn = attn.max(dim=-1).values
                stats['structure_attention_max_sum'] += float(max_attn.sum().item())
                stats['structure_attention_max_count'] += int(max_attn.numel())
        if 'structure_valid_context' in weights and weights['structure_valid_context'] is not None:
            valid = weights['structure_valid_context'].detach()
            stats['structure_valid_context_sum'] += float(valid.sum().item())
            stats['structure_valid_context_count'] += int(valid.numel())
        if 'memory_gate' in weights and weights['memory_gate'] is not None:
            gate = weights['memory_gate'].detach()
            stats['memory_gate_sum'] += float(gate.sum().item())
            stats['memory_gate_count'] += int(gate.numel())
        if 'memory_attention' in weights and weights['memory_attention'] is not None:
            attn = weights['memory_attention'].detach()
            if attn.numel() > 0:
                max_attn = attn.max(dim=-1).values
                stats['memory_attention_max_sum'] += float(max_attn.sum().item())
                stats['memory_attention_max_count'] += int(max_attn.numel())
        if 'memory_valid_slots' in weights and weights['memory_valid_slots'] is not None:
            valid = weights['memory_valid_slots'].detach()
            stats['memory_valid_slots_sum'] += float(valid.sum().item())
            stats['memory_valid_slots_count'] += int(valid.numel())
        rag_info = output.get('rag_info') or {}
        if 'rag_gate' in rag_info and rag_info['rag_gate'] is not None:
            gate = rag_info['rag_gate'].detach()
            stats['rag_gate_sum'] += float(gate.sum().item())
            stats['rag_gate_count'] += int(gate.numel())
        if 'attn_weights' in rag_info and rag_info['attn_weights'] is not None:
            attn = rag_info['attn_weights'].detach()
            if attn.numel() > 0:
                stats['rag_attention_max_sum'] += float(attn.max(dim=-1).values.sum().item())
                stats['rag_attention_max_count'] += int(attn.size(0))

    def _finalize_attention_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        pairs = [
            ('fusion_gate_mean', 'fusion_gate_sum', 'fusion_gate_count'),
            ('projection_cosine_mean', 'projection_cosine_sum', 'projection_cosine_count'),
            ('projection_text_norm_mean', 'projection_text_norm_sum', 'projection_text_norm_count'),
            ('projection_video_norm_mean', 'projection_video_norm_sum', 'projection_video_norm_count'),
            ('disagreement_gate_mean', 'disagreement_gate_sum', 'disagreement_gate_count'),
            ('js_divergence_mean', 'js_divergence_sum', 'js_divergence_count'),
            ('kl_text_video_mean', 'kl_text_video_sum', 'kl_text_video_count'),
            ('kl_video_text_mean', 'kl_video_text_sum', 'kl_video_text_count'),
            ('structure_gate_mean', 'structure_gate_sum', 'structure_gate_count'),
            ('structure_attention_max_mean', 'structure_attention_max_sum', 'structure_attention_max_count'),
            ('structure_valid_context_mean', 'structure_valid_context_sum', 'structure_valid_context_count'),
            ('memory_gate_mean', 'memory_gate_sum', 'memory_gate_count'),
            ('memory_attention_max_mean', 'memory_attention_max_sum', 'memory_attention_max_count'),
            ('memory_valid_slots_mean', 'memory_valid_slots_sum', 'memory_valid_slots_count'),
            ('rag_gate_mean', 'rag_gate_sum', 'rag_gate_count'),
            ('rag_attention_max_mean', 'rag_attention_max_sum', 'rag_attention_max_count'),
        ]
        for name, sum_key, count_key in pairs:
            count = stats.get(count_key, 0)
            if count:
                result[name] = stats[sum_key] / count
        return result

    def _maybe_mask_training_video(self, video_feat: torch.Tensor) -> torch.Tensor:
        mask_prob = float(getattr(self.args, 'train_video_mask_prob', 0.0))
        allow_all_fusions = bool(
            getattr(self.args, 'train_video_mask_apply_all_fusions', False)
        )
        if (
            mask_prob <= 0.0
            or not self.model.training
            or (
                getattr(self.args, 'fusion', None) != 'tv_disagreement'
                and not allow_all_fusions
            )
        ):
            return video_feat
        keep = torch.rand(video_feat.size(0), 1, device=video_feat.device) >= mask_prob
        return video_feat * keep.to(video_feat.dtype)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        attention_stats = {
            'fusion_gate_sum': 0.0, 'fusion_gate_count': 0,
            'projection_cosine_sum': 0.0, 'projection_cosine_count': 0,
            'projection_text_norm_sum': 0.0, 'projection_text_norm_count': 0,
            'projection_video_norm_sum': 0.0, 'projection_video_norm_count': 0,
            'disagreement_gate_sum': 0.0, 'disagreement_gate_count': 0,
            'js_divergence_sum': 0.0, 'js_divergence_count': 0,
            'kl_text_video_sum': 0.0, 'kl_text_video_count': 0,
            'kl_video_text_sum': 0.0, 'kl_video_text_count': 0,
            'structure_gate_sum': 0.0, 'structure_gate_count': 0,
            'structure_attention_max_sum': 0.0, 'structure_attention_max_count': 0,
            'structure_valid_context_sum': 0.0, 'structure_valid_context_count': 0,
            'memory_gate_sum': 0.0, 'memory_gate_count': 0,
            'memory_attention_max_sum': 0.0, 'memory_attention_max_count': 0,
            'memory_valid_slots_sum': 0.0, 'memory_valid_slots_count': 0,
            'rag_gate_sum': 0.0, 'rag_gate_count': 0,
            'rag_attention_max_sum': 0.0, 'rag_attention_max_count': 0,
        }
        
        for batch in dataloader:
            video_feat = batch['video_feat'].to(self.device)
            text_feat = batch['text_feat'].to(self.device)
            labels = batch['label'].to(self.device)
            context_emotions = batch['context_emotions'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            context_kwargs = self._context_kwargs(batch)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if isinstance(self.model, EmotionStyleModel):
                output = self.model(
                    video_feat=video_feat if self.args.fusion not in ('text_only',) else None,
                    text_feat=text_feat if self.args.fusion not in ('video_only',) else None,
                    context_emotions=context_emotions,
                    **context_kwargs,
                )
                # 情感分类损失
                loss_emotion = self.criterion_emotion(output['logits'], labels)
                # 风格预测损失
                style_targets = generate_style_labels(labels).to(self.device)
                loss_style = self.criterion_style(output['style_params'], style_targets)
                # 联合损失
                loss = loss_emotion + self.args.style_loss_weight * loss_style
            elif isinstance(self.model, EmotionRAGClassifier):
                # RAG 模式
                rag_kwargs = {}
                if 'retrieved_features' in batch:
                    rag_kwargs['retrieved_features'] = batch['retrieved_features'].to(self.device)
                    rag_kwargs['retrieved_labels'] = batch['retrieved_labels'].to(self.device)
                    rag_kwargs['retrieved_scores'] = batch['retrieved_scores'].to(self.device)
                output = self.model(video_feat, text_feat, **rag_kwargs)
                loss = self.criterion_emotion(output['logits'], labels)
            else:
                video_input = self._maybe_mask_training_video(video_feat)
                output = self.model(
                    video_feat=video_input if self.args.fusion not in ('text_only',) else None,
                    text_feat=text_feat if self.args.fusion not in ('video_only',) else None,
                    **context_kwargs,
                    return_attention=True,
                )
                loss = self.criterion_emotion(output['logits'], labels)
                aux_weight = float(getattr(self.args, 'projection_aux_weight', 0.0))
                if aux_weight > 0.0 and 'text_logits' in output and 'video_logits' in output:
                    loss = loss + aux_weight * 0.5 * (
                        self.criterion_emotion(output['text_logits'], labels)
                        + self.criterion_emotion(output['video_logits'], labels)
                    )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = output['logits'].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1
            self._accumulate_attention_stats(output, attention_stats)
        
        avg_loss = total_loss / max(num_batches, 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        result = {'loss': avg_loss, 'acc': acc, 'f1': f1, 'macro_f1': macro_f1}
        result.update(self._finalize_attention_stats(attention_stats))
        return result
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证集评估"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0
        attention_stats = {
            'fusion_gate_sum': 0.0, 'fusion_gate_count': 0,
            'projection_cosine_sum': 0.0, 'projection_cosine_count': 0,
            'projection_text_norm_sum': 0.0, 'projection_text_norm_count': 0,
            'projection_video_norm_sum': 0.0, 'projection_video_norm_count': 0,
            'disagreement_gate_sum': 0.0, 'disagreement_gate_count': 0,
            'js_divergence_sum': 0.0, 'js_divergence_count': 0,
            'kl_text_video_sum': 0.0, 'kl_text_video_count': 0,
            'kl_video_text_sum': 0.0, 'kl_video_text_count': 0,
            'structure_gate_sum': 0.0, 'structure_gate_count': 0,
            'structure_attention_max_sum': 0.0, 'structure_attention_max_count': 0,
            'structure_valid_context_sum': 0.0, 'structure_valid_context_count': 0,
            'memory_gate_sum': 0.0, 'memory_gate_count': 0,
            'memory_attention_max_sum': 0.0, 'memory_attention_max_count': 0,
            'memory_valid_slots_sum': 0.0, 'memory_valid_slots_count': 0,
            'rag_gate_sum': 0.0, 'rag_gate_count': 0,
            'rag_attention_max_sum': 0.0, 'rag_attention_max_count': 0,
        }
        
        for batch in dataloader:
            video_feat = batch['video_feat'].to(self.device)
            text_feat = batch['text_feat'].to(self.device)
            labels = batch['label'].to(self.device)
            context_emotions = batch['context_emotions'].to(self.device)
            context_mask = batch['context_mask'].to(self.device)
            context_kwargs = self._context_kwargs(batch)
            
            if isinstance(self.model, EmotionStyleModel):
                output = self.model(
                    video_feat=video_feat if self.args.fusion not in ('text_only',) else None,
                    text_feat=text_feat if self.args.fusion not in ('video_only',) else None,
                    context_emotions=context_emotions,
                    **context_kwargs,
                )
                loss_emotion = self.criterion_emotion(output['logits'], labels)
                style_targets = generate_style_labels(labels).to(self.device)
                loss_style = self.criterion_style(output['style_params'], style_targets)
                loss = loss_emotion + self.args.style_loss_weight * loss_style
            elif isinstance(self.model, EmotionRAGClassifier):
                rag_kwargs = {}
                if 'retrieved_features' in batch:
                    rag_kwargs['retrieved_features'] = batch['retrieved_features'].to(self.device)
                    rag_kwargs['retrieved_labels'] = batch['retrieved_labels'].to(self.device)
                    rag_kwargs['retrieved_scores'] = batch['retrieved_scores'].to(self.device)
                output = self.model(video_feat, text_feat, **rag_kwargs)
                loss = self.criterion_emotion(output['logits'], labels)
            else:
                output = self.model(
                    video_feat=video_feat if self.args.fusion not in ('text_only',) else None,
                    text_feat=text_feat if self.args.fusion not in ('video_only',) else None,
                    **context_kwargs,
                    return_attention=True,
                )
                loss = self.criterion_emotion(output['logits'], labels)
                aux_weight = float(getattr(self.args, 'projection_aux_weight', 0.0))
                if aux_weight > 0.0 and 'text_logits' in output and 'video_logits' in output:
                    loss = loss + aux_weight * 0.5 * (
                        self.criterion_emotion(output['text_logits'], labels)
                        + self.criterion_emotion(output['video_logits'], labels)
                    )
            
            total_loss += loss.item()
            preds = output['logits'].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1
            self._accumulate_attention_stats(output, attention_stats)
        
        avg_loss = total_loss / max(num_batches, 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        result = {'loss': avg_loss, 'acc': acc, 'f1': f1, 'macro_f1': macro_f1}
        result.update(self._finalize_attention_stats(attention_stats))
        
        # 详细报告
        if len(set(all_labels)) > 1:
            try:
                target_names = DATASET_LABELS.get(self.args.dataset)
                report = classification_report(
                    all_labels, all_preds,
                    target_names=target_names,
                    digits=4, zero_division=0,
                    output_dict=True
                )
                self._last_report = report
            except Exception as e:
                print(f"Warning: classification_report failed: {e}")
        
        return result
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: str) -> Dict:
        """完整训练循环"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始训练: fusion={self.args.fusion}, epochs={self.args.epochs}")
        print(f"学习率={self.args.lr}, batch_size={self.args.batch_size}")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['train_macro_f1'].append(train_metrics['macro_f1'])
            for key, value in train_metrics.items():
                if key not in ('loss', 'acc', 'f1', 'macro_f1'):
                    self.history.setdefault(f'train_{key}', []).append(value)
            
            # 验证
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])
            for key, value in val_metrics.items():
                if key not in ('loss', 'acc', 'f1', 'macro_f1'):
                    self.history.setdefault(f'val_{key}', []).append(value)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 打印
            extra_parts = []
            if 'structure_gate_mean' in val_metrics:
                extra_parts.append(f"StructGate={val_metrics['structure_gate_mean']:.4f}")
            if 'structure_attention_max_mean' in val_metrics:
                extra_parts.append(f"StructAttnMax={val_metrics['structure_attention_max_mean']:.4f}")
            if 'fusion_gate_mean' in val_metrics:
                extra_parts.append(f"FusionGate={val_metrics['fusion_gate_mean']:.4f}")
            if 'projection_cosine_mean' in val_metrics:
                extra_parts.append(f"ProjCos={val_metrics['projection_cosine_mean']:.4f}")
            if 'js_divergence_mean' in val_metrics:
                extra_parts.append(f"JS={val_metrics['js_divergence_mean']:.4f}")
            if 'disagreement_gate_mean' in val_metrics:
                extra_parts.append(f"DisGate={val_metrics['disagreement_gate_mean']:.4f}")
            if 'memory_gate_mean' in val_metrics:
                extra_parts.append(f"MemoryGate={val_metrics['memory_gate_mean']:.4f}")
            if 'memory_attention_max_mean' in val_metrics:
                extra_parts.append(f"MemoryAttnMax={val_metrics['memory_attention_max_mean']:.4f}")
            if 'rag_gate_mean' in val_metrics:
                extra_parts.append(f"RagGate={val_metrics['rag_gate_mean']:.4f}")
            if 'rag_attention_max_mean' in val_metrics:
                extra_parts.append(f"RagAttnMax={val_metrics['rag_attention_max_mean']:.4f}")
            extra = " | " + " ".join(extra_parts) if extra_parts else ""
            print(f"Epoch {epoch:3d}/{self.args.epochs} | "
                  f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} "
                  f"W-F1={train_metrics['f1']:.4f} M-F1={train_metrics['macro_f1']:.4f} | "
                  f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} "
                  f"W-F1={val_metrics['f1']:.4f} M-F1={val_metrics['macro_f1']:.4f} | "
                  f"lr={current_lr:.6f}{extra}")
            
            # 保存最佳模型
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'val_acc': val_metrics['acc'],
                    'val_macro_f1': val_metrics['macro_f1'],
                    'args': vars(self.args),
                }, checkpoint_path)
                print(f"  ⭐ 最佳模型已保存 (F1={val_metrics['f1']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    print(f"\n⏹ 早停: {self.args.patience} 个epoch无提升")
                    break
        
        print(f"\n{'='*60}")
        print(f"训练完成! 最佳epoch={self.best_epoch}, 最佳val F1={self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        # 保存训练历史
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        best_idx = max(self.best_epoch - 1, 0)
        result = {
            'best_epoch': self.best_epoch,
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.history['val_acc'][self.best_epoch - 1],
            'best_val_macro_f1': self.history['val_macro_f1'][self.best_epoch - 1],
            'history': self.history,
        }
        for key, values in self.history.items():
            if key.startswith('val_') and key not in ('val_loss', 'val_acc', 'val_f1', 'val_macro_f1'):
                if best_idx < len(values):
                    result[f'best_{key}'] = values[best_idx]
        return result


# ===================== 主函数 =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args) -> nn.Module:
    """根据参数构建模型"""
    rag_mode = getattr(args, 'rag_mode', 'none')
    
    if rag_mode != 'none':
        # Emotion-RAG 模型
        model = EmotionRAGClassifier(
            video_dim=args.feat_dim,
            text_dim=args.feat_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes if hasattr(args, 'num_classes') and args.num_classes else 7,
            dropout=args.dropout,
            num_heads=args.num_heads,
            top_k=getattr(args, 'rag_top_k', 5),
        )
        return model
    
    # 情感分类器
    classifier = EmotionClassifier(
        video_dim=args.feat_dim,
        text_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes if hasattr(args, 'num_classes') and args.num_classes else 7,
        fusion_type=args.fusion,
        dropout=args.dropout,
        num_heads=args.num_heads,
        structure_mode=getattr(args, 'structure_mode', 'none'),
        max_context_len=getattr(args, 'context_len', 5),
        max_distance=getattr(args, 'context_max_distance', 20),
        num_relations=len(RELATION_LABELS),
        structure_gate_scale=getattr(args, 'structure_gate_scale', 1.0),
        relation_dropout=getattr(args, 'relation_dropout', 0.0),
        relation_embedding_init_std=getattr(args, 'relation_embedding_init_std', None),
        speaker_memory_mode=getattr(args, 'speaker_memory_mode', 'none'),
        speaker_memory_slots=getattr(args, 'speaker_memory_slots', 4),
        memory_gate_scale=getattr(args, 'memory_gate_scale', 1.0),
        disagreement_gate_min=getattr(args, 'disagreement_gate_min', 0.0),
        disagreement_gate_temperature=getattr(args, 'disagreement_gate_temperature', 1.0),
        disagreement_gate_bias_init=getattr(args, 'disagreement_gate_bias_init', 0.0),
    )
    
    if args.with_style:
        # 联合模型：分类器 + 风格预测
        style_pred = StylePredictor(
            num_emotions=7,
            emotion_embed_dim=64,
            context_dim=64,
            style_dim=3,
            use_emotion_features=True,
            feature_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        model = EmotionStyleModel(classifier, style_pred)
    else:
        model = classifier
    
    return model


def _build_rag_cache(model, train_dataset, val_dataset, args, device):
    """
    构建 RAG 记忆库并预计算所有样本的检索结果
    
    流程：
    1. 用模型的 fusion 层计算训练集所有样本的融合特征
    2. 用这些融合特征构建 FAISS 索引（记忆库）
    3. 对训练集和验证集的每个样本，按 rag_mode 检索 Top-K
    4. 将检索结果缓存到 dataset._rag_cache 中
    """
    model.eval()
    model_on_device = model.to(device)
    
    # Step 1: 计算训练集融合特征
    print("  [1/3] 计算训练集融合特征...")
    all_fused = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(train_dataset), batch_size):
            end = min(i + batch_size, len(train_dataset))
            v_batch = train_dataset.video_features[i:end].to(device)
            t_batch = train_dataset.text_features[i:end].to(device)
            fused = model_on_device.fusion(v_batch, t_batch)
            all_fused.append(fused.cpu().numpy())
    
    train_fused = np.concatenate(all_fused, axis=0).astype(np.float32)  # [N_train, hidden_dim]
    train_labels = train_dataset.labels.numpy().astype(np.int64)
    print(f"    融合特征: {train_fused.shape}")
    
    # Step 2: 构建记忆库
    print("  [2/3] 构建 FAISS 记忆库...")
    memory_bank = EmotionMemoryBank(train_fused.shape[1])
    memory_bank.build(
        features=train_fused,
        labels=train_labels,
        speaker_ids=train_dataset.speaker_ids,
        dialogue_ids=train_dataset.dialogue_ids,
        utterance_ids=train_dataset.utterance_ids,
    )
    
    # Step 3: 预计算检索结果
    print(f"  [3/3] 预计算 RAG 检索 (mode={args.rag_mode}, K={args.rag_top_k})...")
    
    rag_mode = args.rag_mode
    top_k = args.rag_top_k
    filter_mode = getattr(args, 'rag_filter_mode', 'raw')
    min_similarity = float(getattr(args, 'rag_min_similarity', -1.0))
    label_consistency = float(getattr(args, 'rag_label_consistency', 0.0))
    fallback_count = 0  # 统计 fallback 到 global 的次数
    filter_stats = {
        'mode': filter_mode,
        'min_similarity': min_similarity,
        'label_consistency': label_consistency,
        'train_raw_candidates': 0,
        'train_kept_candidates': 0,
        'val_raw_candidates': 0,
        'val_kept_candidates': 0,
        'train_empty_after_filter': 0,
        'val_empty_after_filter': 0,
        'train_self_retrieval_excluded': True,
    }

    def _apply_rag_filter(feats, labels, scores, split_name: str):
        raw_valid = int(np.sum(labels >= 0))
        filter_stats[f'{split_name}_raw_candidates'] += raw_valid
        if filter_mode == 'raw':
            filter_stats[f'{split_name}_kept_candidates'] += raw_valid
            return feats, labels, scores

        keep = (labels >= 0) & (scores >= min_similarity)
        valid_labels = labels[keep]
        if label_consistency > 0.0 and len(valid_labels) > 0:
            unique, counts = np.unique(valid_labels, return_counts=True)
            majority_label = unique[int(np.argmax(counts))]
            majority_ratio = float(np.max(counts) / len(valid_labels))
            if majority_ratio >= label_consistency:
                keep = keep & (labels == majority_label)
            else:
                keep = np.zeros_like(keep, dtype=bool)

        kept_indices = np.where(keep)[0]
        kept = int(len(kept_indices))
        filter_stats[f'{split_name}_kept_candidates'] += kept
        if kept == 0:
            filter_stats[f'{split_name}_empty_after_filter'] += 1
            return (
                np.zeros_like(feats, dtype=np.float32),
                np.full_like(labels, -1, dtype=np.int64),
                np.zeros_like(scores, dtype=np.float32),
            )

        filtered_feats = np.zeros_like(feats, dtype=np.float32)
        filtered_labels = np.full_like(labels, -1, dtype=np.int64)
        filtered_scores = np.zeros_like(scores, dtype=np.float32)
        slot_count = min(top_k, kept)
        filtered_feats[:slot_count] = feats[kept_indices[:slot_count]]
        filtered_labels[:slot_count] = labels[kept_indices[:slot_count]]
        filtered_scores[:slot_count] = scores[kept_indices[:slot_count]]
        return filtered_feats, filtered_labels, filtered_scores
    
    def _retrieve_with_fallback(query, idx, dataset, exclude_idx=None):
        """检索 + 不足时用 Global 补齐"""
        nonlocal fallback_count
        
        if rag_mode == 'global':
            return memory_bank.retrieve_global(query, top_k, exclude_idx=exclude_idx)
        
        # 先用指定模式检索
        if rag_mode == 'speaker':
            sid = dataset.speaker_ids[idx] if (dataset.speaker_ids is not None and idx < len(dataset.speaker_ids)) else None
            if sid is not None:
                feats, labels, scores = memory_bank.retrieve_by_speaker(query, sid, top_k, exclude_idx=exclude_idx)
            else:
                return memory_bank.retrieve_global(query, top_k, exclude_idx=exclude_idx)
        elif rag_mode == 'dialogue':
            did = dataset.dialogue_ids[idx] if (dataset.dialogue_ids is not None and idx < len(dataset.dialogue_ids)) else None
            uid = dataset.utterance_ids[idx] if (dataset.utterance_ids is not None and idx < len(dataset.utterance_ids)) else None
            if did is not None:
                feats, labels, scores = memory_bank.retrieve_by_dialogue(query, did, uid, top_k, exclude_idx=exclude_idx)
            else:
                return memory_bank.retrieve_global(query, top_k, exclude_idx=exclude_idx)
        else:
            return memory_bank.retrieve_global(query, top_k, exclude_idx=exclude_idx)
        
        # 检查是否有 padding 槽位（label == -1），用 Global 补齐
        padding_mask = (labels == -1)
        num_padding = int(padding_mask.sum())
        if num_padding > 0:
            fallback_count += 1
            # Global 检索足够多的候选
            g_feats, g_labels, g_scores = memory_bank.retrieve_global(query, top_k + num_padding, exclude_idx=exclude_idx)
            # 从 global 结果中取出 padding 数量的候选来补齐
            fill_idx = 0
            for i in range(top_k):
                if padding_mask[i]:
                    while fill_idx < len(g_labels) and g_labels[fill_idx] in labels[:i].tolist():
                        fill_idx += 1  # 跳过已有的
                    if fill_idx < len(g_labels):
                        feats[i] = g_feats[fill_idx]
                        labels[i] = g_labels[fill_idx]
                        scores[i] = g_scores[fill_idx]
                        fill_idx += 1
        
        return feats, labels, scores
    
    # 训练集检索
    train_cache = {}
    hit_count = 0
    for idx in range(len(train_dataset)):
        query = train_fused[idx]
        feats, labels, scores = _retrieve_with_fallback(query, idx, train_dataset, exclude_idx=idx)
        
        feats, labels, scores = _apply_rag_filter(feats, labels, scores, 'train')
        train_cache[idx] = (feats, labels, scores)
        current_label = train_labels[idx]
        hit_count += np.sum(labels == current_label)
    
    train_dataset._rag_cache = train_cache
    denom = max(filter_stats['train_kept_candidates'], 1)
    hit_rate = hit_count / denom * 100
    train_retention = filter_stats['train_kept_candidates'] / max(filter_stats['train_raw_candidates'], 1) * 100
    print(f"    训练集: {len(train_cache)} 条缓存, 同标签命中率={hit_rate:.1f}%, global fallback={fallback_count} 次")
    print(f"    训练集 RAG 过滤: retained={train_retention:.1f}%, empty_after_filter={filter_stats['train_empty_after_filter']}")
    
    # 验证集检索（从训练集记忆库中检索，不排除自身）
    val_cache = {}
    val_hit_count = 0
    fallback_count = 0  # 重置计数
    with torch.no_grad():
        val_fused_list = []
        batch_size = 256
        for i in range(0, len(val_dataset), batch_size):
            end = min(i + batch_size, len(val_dataset))
            v_batch = val_dataset.video_features[i:end].to(device)
            t_batch = val_dataset.text_features[i:end].to(device)
            fused = model_on_device.fusion(v_batch, t_batch)
            val_fused_list.append(fused.cpu().numpy())
    val_fused = np.concatenate(val_fused_list, axis=0).astype(np.float32)
    val_labels = val_dataset.labels.numpy().astype(np.int64)
    
    for idx in range(len(val_dataset)):
        query = val_fused[idx]
        feats, labels, scores = _retrieve_with_fallback(query, idx, val_dataset, exclude_idx=None)
        
        feats, labels, scores = _apply_rag_filter(feats, labels, scores, 'val')
        val_cache[idx] = (feats, labels, scores)
        current_label = val_labels[idx]
        val_hit_count += np.sum(labels == current_label)
    
    val_dataset._rag_cache = val_cache
    val_denom = max(filter_stats['val_kept_candidates'], 1)
    val_hit_rate = val_hit_count / val_denom * 100
    val_retention = filter_stats['val_kept_candidates'] / max(filter_stats['val_raw_candidates'], 1) * 100
    print(f"    验证集: {len(val_cache)} 条缓存, 同标签命中率={val_hit_rate:.1f}%, global fallback={fallback_count} 次")
    print(f"    验证集 RAG 过滤: retained={val_retention:.1f}%, empty_after_filter={filter_stats['val_empty_after_filter']}")
    args.rag_filter_stats = filter_stats
    print(f"  ✓ RAG 缓存构建完成")
    
    model.train()  # 恢复训练模式


def main():
    parser = argparse.ArgumentParser(description='多模态情感融合训练')
    
    # 数据
    parser.add_argument('--feature_dir', type=str, default='datasets/MELD/features',
                        help='缓存特征目录')
    parser.add_argument('--smoke_test', action='store_true',
                        help='用随机假数据冒烟测试')
    parser.add_argument('--quick_test', action='store_true',
                        help='用dev数据同时做训练和验证（快速验证真实特征）')
    
    # 模型
    parser.add_argument('--dataset', type=str, default='meld',
                        choices=['meld', 'chsims', 'mosei'],
                        help='使用的数据集')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='如果不指定则自动根据 dataset 设定')
    parser.add_argument('--fusion', type=str, default='attention',
                        choices=['concat', 'attention', 'gated', 'text_only', 'video_only', 'tv_projection', 'tv_disagreement'],
                        help='融合策略')
    parser.add_argument('--feat_dim', type=int, default=3584,
                        help='特征维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐层维度')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比率')
    parser.add_argument('--with_style', action='store_true',
                        help='是否联合训练风格预测')
    parser.add_argument('--style_loss_weight', type=float, default=0.5,
                        help='风格预测损失权重')
    parser.add_argument('--projection_aux_weight', type=float, default=0.0,
                        help='tv_projection/tv_disagreement 单模态 emotion head 辅助损失权重；0 表示只训练融合分类头')
    parser.add_argument('--train_video_mask_prob', type=float, default=0.0,
                        help='Step 10.5: 训练阶段随机置零 video feature 的概率；仅 tv_disagreement 使用')
    parser.add_argument('--train_video_mask_apply_all_fusions', action='store_true',
                        help='显式允许训练 video masking 作用于非 tv_disagreement 融合；默认关闭以保持历史行为')
    parser.add_argument('--disagreement_gate_min', type=float, default=0.0,
                        help='Step 10.5: disagreement gate 下限，0 表示保持原行为')
    parser.add_argument('--disagreement_gate_temperature', type=float, default=1.0,
                        help='Step 10.5: disagreement gate sigmoid 温度')
    parser.add_argument('--disagreement_gate_bias_init', type=float, default=0.0,
                        help='Step 10.5: disagreement gate 最后一层 bias 初始化')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2正则化')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='兼容旧接口：等价于 --loss_type focal')
    parser.add_argument('--loss_type', choices=['ce', 'focal'], default='ce',
                        help='情感分类损失类型')
    parser.add_argument('--class_weight_mode',
                        choices=['none', 'inverse', 'sqrt_inverse'], default='none',
                        help='类别权重；仅根据 train split 动态统计')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--context_len', type=int, default=5,
                        help='对话上下文窗口')
    parser.add_argument('--structure_mode', type=str, default='none',
                        choices=['none', 'context', 'speaker_distance', 'validated_relations'],
                        help='结构上下文模式：none=不用上下文，context=前K轮特征，speaker_distance=前K轮特征+same-speaker/distance，validated_relations=再加入Step4 relation id')
    parser.add_argument('--context_max_distance', type=int, default=20,
                        help='turn distance embedding 最大距离')
    parser.add_argument('--validated_relations_path', type=str, default=None,
                        help='Step 4 输出的 validated_relations.jsonl')
    parser.add_argument('--structure_gate_scale', type=float, default=1.0,
                        help='结构上下文 gate 缩放；1.0 保持原 Step 5 行为')
    parser.add_argument('--relation_dropout', type=float, default=0.0,
                        help='speaker/distance/relation embedding dropout')
    parser.add_argument('--relation_embedding_init_std', type=float, default=None,
                        help='relation embedding 小方差初始化标准差；默认 None 使用 PyTorch 初始化')
    parser.add_argument('--speaker_memory_mode', type=str, default='none',
                        choices=['none', 'prototype'],
                        help='speaker memory 模式：none=关闭，prototype=固定离线聚类原型')
    parser.add_argument('--speaker_memory_slots', type=int, default=4,
                        help='每个 speaker 的固定 prototype slot 数')
    parser.add_argument('--memory_gate_scale', type=float, default=1.0,
                        help='speaker memory gate 缩放')
    parser.add_argument('--speaker_prototype_path', type=str, default=None,
                        help='speaker prototype npz 路径；未提供时在输出目录自动生成')
    parser.add_argument('--augmentation_bundle', type=str, default=None,
                        help='Step 19 train-only 增强 bundle (.npz)；仅追加 current utterance')
    
    # RAG
    parser.add_argument('--rag_mode', type=str, default='none',
                        choices=['none', 'global', 'speaker', 'dialogue'],
                        help='RAG 检索模式')
    parser.add_argument('--rag_top_k', type=int, default=5,
                        help='RAG 检索 Top-K 数量')
    parser.add_argument('--rag_filter_mode', type=str, default='raw',
                        choices=['raw', 'filtered'],
                        help='raw=不过滤检索候选；filtered=启用 Step 8 相似度和标签一致性过滤')
    parser.add_argument('--rag_min_similarity', type=float, default=-1.0,
                        help='filtered RAG 的最小余弦相似度阈值')
    parser.add_argument('--rag_label_consistency', type=float, default=0.0,
                        help='filtered RAG 的候选标签多数一致性阈值；0 表示不启用')
    
    # 其他
    parser.add_argument('--output_dir', type=str, default='outputs/training',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 处理相对路径
    project_root = Path(__file__).parent.parent.parent  # src/training/ -> src/ -> project root
    if not os.path.isabs(args.feature_dir):
        args.feature_dir = str(project_root / args.feature_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = str(project_root / args.output_dir)
    if args.speaker_prototype_path and not os.path.isabs(args.speaker_prototype_path):
        args.speaker_prototype_path = str(project_root / args.speaker_prototype_path)
    if args.augmentation_bundle and not os.path.isabs(args.augmentation_bundle):
        args.augmentation_bundle = str(project_root / args.augmentation_bundle)
    
    # 加载数据
    if args.smoke_test:
        train_data = create_smoke_test_data(200, args.feat_dim)
        val_data = create_smoke_test_data(50, args.feat_dim)
    else:
        print("加载特征...")
        val_split = 'dev' if os.path.exists(os.path.join(args.feature_dir, 'dev_video_features.npy')) else 'valid'
        
        if args.quick_test:
            # 快速验证模式：用 val_split 数据同时做训练和验证
            print(f"⚡ 快速验证模式：使用 {val_split} 数据")
            train_data = load_cached_features(args.feature_dir, val_split)
            val_data = train_data  # 训练和验证用同一份数据
        else:
            train_data = load_cached_features(args.feature_dir, 'train')
            val_data = load_cached_features(args.feature_dir, val_split)
        
        # 加载 speaker 信息（RAG-Speaker 模式需要）
        for split_name, split_data in [('train', train_data), (val_split, val_data)]:
            speaker_path = os.path.join(args.feature_dir, f'{split_name}_speakers.npy')
            if os.path.exists(speaker_path):
                split_data['speakers'] = np.load(speaker_path, allow_pickle=True)
                print(f"📌 加载 speaker 信息: {speaker_path} ({len(split_data['speakers'])} 条)")

        if args.augmentation_bundle:
            bundle = np.load(args.augmentation_bundle, allow_pickle=False)
            source_indices = np.asarray(bundle['source_indices'], dtype=np.int64)
            augmented_text = np.asarray(bundle['text_features'], dtype=np.float32)
            original_count = len(train_data['labels'])
            if augmented_text.ndim != 2 or augmented_text.shape[1] != train_data['text_features'].shape[1]:
                raise ValueError('augmentation text feature shape mismatch')
            if len(source_indices) != len(augmented_text) or np.any(source_indices < 0) or np.any(source_indices >= original_count):
                raise ValueError('augmentation source_indices are invalid')
            for key in ['video_features', 'labels', 'dialogue_ids', 'utterance_ids', 'speakers']:
                if key in train_data:
                    train_data[key] = np.concatenate([train_data[key], train_data[key][source_indices]], axis=0)
            train_data['text_features'] = np.concatenate([train_data['text_features'], augmented_text], axis=0)
            train_data['is_augmented'] = np.concatenate([
                np.zeros(original_count, dtype=bool), np.ones(len(source_indices), dtype=bool)
            ])
            train_data['source_indices'] = np.concatenate([
                np.arange(original_count, dtype=np.int64), source_indices
            ])
            print(f"📌 Step 19 train-only augmentation: +{len(source_indices)} current utterances from {args.augmentation_bundle}")

        if args.validated_relations_path:
            for split_name, split_data in [('train', train_data), (val_split, val_data)]:
                split_data['relation_ids'] = load_validated_relation_ids(
                    args.validated_relations_path,
                    split_name,
                    len(split_data['labels']),
                )
                unique_rel, rel_counts = np.unique(split_data['relation_ids'], return_counts=True)
                rel_dist = {
                    RELATION_LABELS.get(int(rel), str(rel)): int(count)
                    for rel, count in zip(unique_rel, rel_counts)
                }
                print(f"📌 加载 validated relations: split={split_name}, {rel_dist}")
        
        # 自动检测特征维度（支持不同模型：Qwen=3584, LLaVA=4096）
        detected_dim = train_data['video_features'].shape[1]
        if args.feat_dim != detected_dim:
            print(f"📌 自动调整 feat_dim: {args.feat_dim} → {detected_dim}")
            args.feat_dim = detected_dim
            
        # 自动设定分类数
        if args.num_classes is None:
            args.num_classes = len(DATASET_LABELS[args.dataset])
            print(f"📌 自动设定 num_classes = {args.num_classes} ({args.dataset})")
            
        unique, counts = np.unique(train_data['labels'], return_counts=True)
        args.label_counts = dict(zip(unique.tolist(), counts.tolist()))
        print(f"📌 自动统计类别分布: {args.label_counts}")

    speaker_prototypes = None
    if args.speaker_memory_mode == 'prototype':
        if args.speaker_prototype_path and os.path.exists(args.speaker_prototype_path):
            speaker_prototypes = load_speaker_prototypes(args.speaker_prototype_path)
            print(f"📌 加载 speaker prototypes: {args.speaker_prototype_path} ({len(speaker_prototypes)} speakers)")
        else:
            original_mask = ~train_data.get('is_augmented', np.zeros(len(train_data['labels']), dtype=bool))
            speaker_prototypes = build_speaker_prototypes(
                train_data['video_features'][original_mask],
                train_data['text_features'][original_mask],
                train_data.get('speakers')[original_mask] if train_data.get('speakers') is not None else None,
                slots=args.speaker_memory_slots,
                seed=args.seed,
            )
            prototype_path = args.speaker_prototype_path or os.path.join(args.output_dir, "speaker_prototypes.npz")
            save_speaker_prototypes(prototype_path, speaker_prototypes)
            args.speaker_prototype_path = prototype_path
            non_global = [k for k in speaker_prototypes if k != "__GLOBAL__"]
            slot_counts = [
                int(speaker_prototypes[k]["mask"].sum())
                for k in non_global
            ]
            print(
                f"📌 生成 speaker prototypes: {prototype_path} "
                f"speakers={len(non_global)}, avg_slots={np.mean(slot_counts) if slot_counts else 0:.2f}"
            )
    
    # 创建Dataset
    train_dataset = MELDFeatureDataset(
        train_data['video_features'], train_data['text_features'],
        train_data['labels'],
        train_data.get('dialogue_ids'), train_data.get('utterance_ids'),
        speaker_ids=train_data.get('speakers'),
        relation_ids=train_data.get('relation_ids'),
        is_augmented=train_data.get('is_augmented'),
        source_indices=train_data.get('source_indices'),
        context_len=args.context_len,
        rag_mode=args.rag_mode, rag_top_k=args.rag_top_k,
        context_max_distance=args.context_max_distance,
        speaker_prototypes=speaker_prototypes,
        speaker_memory_slots=args.speaker_memory_slots,
    )
    val_dataset = MELDFeatureDataset(
        val_data['video_features'], val_data['text_features'],
        val_data['labels'],
        val_data.get('dialogue_ids'), val_data.get('utterance_ids'),
        speaker_ids=val_data.get('speakers'),
        relation_ids=val_data.get('relation_ids'),
        context_len=args.context_len,
        rag_mode=args.rag_mode, rag_top_k=args.rag_top_k,
        context_max_distance=args.context_max_distance,
        speaker_prototypes=speaker_prototypes,
        speaker_memory_slots=args.speaker_memory_slots,
    )
    
    # 构建模型（需要在 RAG 预计算之前，因为需要 fusion 层）
    model = build_model(args)
    print(f"模型: {type(model).__name__}")
    print(f"融合方式: {args.fusion}")
    if args.fusion == 'tv_disagreement':
        print(
            "Disagreement fix params: "
            f"train_video_mask_prob={args.train_video_mask_prob}, "
            f"gate_min={args.disagreement_gate_min}, "
            f"gate_temperature={args.disagreement_gate_temperature}, "
            f"gate_bias_init={args.disagreement_gate_bias_init}"
        )
    print(f"结构上下文: {args.structure_mode}, context_len={args.context_len}")
    if args.structure_mode != 'none':
        print(
            "结构修正参数: "
            f"gate_scale={args.structure_gate_scale}, "
            f"relation_dropout={args.relation_dropout}, "
            f"relation_embedding_init_std={args.relation_embedding_init_std}"
        )
    if args.speaker_memory_mode != 'none':
        print(
            "Speaker memory: "
            f"mode={args.speaker_memory_mode}, "
            f"slots={args.speaker_memory_slots}, "
            f"gate_scale={args.memory_gate_scale}, "
            f"prototype_path={args.speaker_prototype_path}"
        )
    if args.rag_mode != 'none':
        print(
            f"RAG 模式: {args.rag_mode}, Top-K={args.rag_top_k}, "
            f"filter={args.rag_filter_mode}, min_sim={args.rag_min_similarity}, "
            f"label_consistency={args.rag_label_consistency}"
        )
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # RAG 模式：构建记忆库 + 预计算检索结果
    if args.rag_mode != 'none':
        print(f"\n🔍 构建 Emotion Memory Bank (RAG mode={args.rag_mode})...")
        _build_rag_cache(model, train_dataset, val_dataset, args, device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rag_tag = f"_rag{args.rag_mode}_k{args.rag_top_k}" if args.rag_mode != 'none' else ""
    save_dir = os.path.join(args.output_dir, f"{args.fusion}{rag_tag}_{timestamp}")
    
    # 训练
    trainer = Trainer(model, args, device)
    results = trainer.train(train_loader, val_loader, save_dir)
    
    # 保存实验配置
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        config = vars(args).copy()
        config['results'] = {
            'best_epoch': results['best_epoch'],
            'best_val_f1': results['best_val_f1'],
            'best_val_acc': results['best_val_acc'],
            'best_val_macro_f1': results['best_val_macro_f1'],
        }
        for key, value in results.items():
            if key.startswith('best_val_') and key not in config['results']:
                config['results'][key] = value
        json.dump(config, f, indent=2)
    
    print(f"\n实验结果保存在: {save_dir}")


if __name__ == "__main__":
    main()
