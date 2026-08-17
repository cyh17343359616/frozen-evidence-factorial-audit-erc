"""
可学习的交互风格预测网络
替代硬编码的情感→交互风格映射表，实现端到端的风格参数预测

特点：
1. 输入情感概率分布（而非离散标签），保留不确定性信息
2. GRU上下文建模，考虑对话历史中的情感演变
3. 注意力机制提供可解释性（哪些历史轮次影响了当前决策）

Author: 陈裕瀚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ===================== 风格参数定义 =====================

@dataclass
class StyleParameters:
    """交互风格的连续参数表示"""
    tone_score: float       # 0=冷静/中性 → 1=热情/活泼
    conciseness_score: float  # 0=简短 → 1=详细
    empathy_score: float    # 0=低共情 → 1=高共情
    
    def to_discrete(self) -> Dict[str, str]:
        """将连续参数离散化为可读的风格描述"""
        # 语气
        if self.tone_score < 0.3:
            tone = 'calm'
        elif self.tone_score < 0.6:
            tone = 'neutral'
        else:
            tone = 'cheerful'
        
        # 简洁度
        if self.conciseness_score < 0.33:
            conciseness = 'brief'
        elif self.conciseness_score < 0.66:
            conciseness = 'medium'
        else:
            conciseness = 'detailed'
        
        # 共情度
        if self.empathy_score < 0.33:
            empathy = 'low'
        elif self.empathy_score < 0.66:
            empathy = 'medium'
        else:
            empathy = 'high'
        
        return {
            'tone': tone,
            'conciseness': conciseness,
            'empathy': empathy,
            'tone_score': round(self.tone_score, 3),
            'conciseness_score': round(self.conciseness_score, 3),
            'empathy_score': round(self.empathy_score, 3),
        }
    
    def get_description(self) -> str:
        """生成可读的风格描述"""
        d = self.to_discrete()
        desc_parts = []
        
        tone_map = {
            'calm': '冷静回应',
            'neutral': '平和回复',
            'cheerful': '活泼积极'
        }
        empathy_map = {
            'low': '保持客观',
            'medium': '适度共情',
            'high': '高度共情'
        }
        concise_map = {
            'brief': '简短精炼',
            'medium': '适中篇幅',
            'detailed': '详细回复'
        }
        
        desc_parts.append(tone_map.get(d['tone'], ''))
        desc_parts.append(empathy_map.get(d['empathy'], ''))
        desc_parts.append(concise_map.get(d['conciseness'], ''))
        
        return '，'.join(desc_parts)


# ===================== 风格标签生成（训练用）=====================

# 基于规则的 soft label（作为训练目标）
# 每种情感对应 [tone, conciseness, empathy] 的连续值
EMOTION_STYLE_SOFT_LABELS = {
    #                 tone  concise  empathy
    'neutral':    [0.5,   0.5,     0.2],
    'joy':        [0.8,   0.5,     0.5],
    'surprise':   [0.6,   0.3,     0.5],
    'fear':       [0.2,   0.3,     0.9],
    'sadness':    [0.3,   0.8,     0.9],
    'anger':      [0.1,   0.3,     0.8],
    'disgust':    [0.4,   0.3,     0.5],
}

EMOTION_NAMES = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']


def generate_style_labels(emotion_labels: torch.Tensor) -> torch.Tensor:
    """
    根据情感标签生成风格参数的训练目标
    
    Args:
        emotion_labels: [B] 情感标签 (0-6)
    
    Returns:
        style_targets: [B, 3] 风格参数目标 (tone, conciseness, empathy)
    """
    label_tensor = torch.tensor(
        [EMOTION_STYLE_SOFT_LABELS[name] for name in EMOTION_NAMES],
        dtype=torch.float32
    ).to(emotion_labels.device)  # [7, 3]
    
    return label_tensor[emotion_labels]  # [B, 3]


# ===================== 风格预测网络 =====================

class StylePredictor(nn.Module):
    """
    可学习的交互风格预测网络
    
    架构：
    1. 情感编码器: 情感概率分布 → 情感嵌入
    2. 上下文GRU: 对话历史中的情感序列 → 上下文表示
    3. 上下文注意力: 当前情感 attend to 历史 → 加权上下文
    4. 风格预测头: 情感嵌入 + 上下文 → 3个连续风格参数
    """
    
    def __init__(self, num_emotions: int = 7, emotion_embed_dim: int = 64,
                 context_dim: int = 64, style_dim: int = 3,
                 num_attn_heads: int = 4, max_context_len: int = 10,
                 dropout: float = 0.1, use_emotion_features: bool = False,
                 feature_dim: int = 256):
        """
        Args:
            num_emotions: 情感类别数
            emotion_embed_dim: 情感嵌入维度
            context_dim: 上下文表示维度  
            style_dim: 风格参数维度（tone, conciseness, empathy）
            num_attn_heads: 注意力头数
            max_context_len: 最大上下文长度
            dropout: Dropout比率
            use_emotion_features: 是否使用融合模型的隐层特征
            feature_dim: 融合特征维度（当 use_emotion_features=True 时）
        """
        super().__init__()
        
        self.num_emotions = num_emotions
        self.style_dim = style_dim
        self.max_context_len = max_context_len
        self.use_emotion_features = use_emotion_features
        
        # 1. 情感编码器
        self.emotion_encoder = nn.Sequential(
            nn.Linear(num_emotions, emotion_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 可选：融合特征编码器
        if use_emotion_features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, emotion_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            input_dim = emotion_embed_dim * 2  # 情感概率 + 融合特征
        else:
            input_dim = emotion_embed_dim
        
        # 2. 上下文GRU
        self.context_gru = nn.GRU(
            input_size=num_emotions,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # 3. 上下文注意力
        self.context_query = nn.Linear(input_dim, context_dim)
        self.context_key = nn.Linear(context_dim, context_dim)
        self.context_value = nn.Linear(context_dim, context_dim)
        
        # 4. 风格预测头
        self.style_head = nn.Sequential(
            nn.Linear(input_dim + context_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, style_dim),
            nn.Sigmoid()  # 输出 [0, 1] 范围
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, emotion_probs: torch.Tensor,
                context_emotions: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                emotion_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            emotion_probs: [B, 7] 当前轮的情感概率分布
            context_emotions: [B, T, 7] 对话历史的情感概率序列
            context_mask: [B, T] 历史轮次的有效掩码 (1=有效, 0=padding)
            emotion_features: [B, D] 融合模型的隐层特征（可选）
        
        Returns:
            dict:
                'style_params': [B, 3] 风格参数 (tone, conciseness, empathy)
                'context_attention': [B, T] 上下文注意力权重（可解释性）
        """
        B = emotion_probs.size(0)
        
        # 1. 编码当前情感
        emotion_embed = self.emotion_encoder(emotion_probs)  # [B, E]
        
        if self.use_emotion_features and emotion_features is not None:
            feat_embed = self.feature_encoder(emotion_features)  # [B, E]
            current_repr = torch.cat([emotion_embed, feat_embed], dim=-1)  # [B, 2E]
        else:
            current_repr = emotion_embed  # [B, E]
        
        # 2. 上下文建模
        if context_emotions is not None and context_emotions.size(1) > 0:
            # 检查是否有有效上下文（避免全mask导致NaN）
            has_valid_context = True
            if context_mask is not None:
                has_valid_context = context_mask.sum(dim=-1).min().item() > 0
            
            if has_valid_context:
                # GRU处理历史序列
                gru_output, _ = self.context_gru(context_emotions)  # [B, T, C]
                
                # 3. 注意力：当前情感 attend to 历史
                query = self.context_query(current_repr).unsqueeze(1)  # [B, 1, C]
                keys = self.context_key(gru_output)                     # [B, T, C]
                values = self.context_value(gru_output)                 # [B, T, C]
                
                # 计算注意力分数
                attn_scores = torch.bmm(query, keys.transpose(1, 2))  # [B, 1, T]
                attn_scores = attn_scores / (keys.size(-1) ** 0.5)
                
                # 应用mask
                if context_mask is not None:
                    mask = context_mask.unsqueeze(1)  # [B, 1, T]
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                
                context_attention = F.softmax(attn_scores, dim=-1)  # [B, 1, T]
                
                # 处理极端情况：如果某些样本全被mask，替换NaN为0
                context_attention = context_attention.nan_to_num(0.0)
                
                context_repr = torch.bmm(context_attention, values).squeeze(1)  # [B, C]
                context_attention = context_attention.squeeze(1)  # [B, T]
            else:
                # 无有效上下文，使用零向量
                context_repr = torch.zeros(B, self.context_gru.hidden_size, 
                                          device=emotion_probs.device)
                context_attention = None
        else:
            # 无上下文时，使用零向量
            context_repr = torch.zeros(B, self.context_gru.hidden_size, 
                                      device=emotion_probs.device)
            context_attention = None
        
        # 4. 预测风格参数
        combined = torch.cat([current_repr, context_repr], dim=-1)
        style_params = self.style_head(combined)  # [B, 3]
        
        return {
            'style_params': style_params,
            'context_attention': context_attention,
        }
    
    def predict_style(self, emotion_probs: torch.Tensor,
                      context_emotions: Optional[torch.Tensor] = None) -> List[StyleParameters]:
        """
        便捷方法：预测并返回StyleParameters对象列表
        
        Args:
            emotion_probs: [B, 7]
            context_emotions: [B, T, 7]
        
        Returns:
            list of StyleParameters
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(emotion_probs, context_emotions)
            params = output['style_params'].cpu().numpy()
        
        return [
            StyleParameters(
                tone_score=float(p[0]),
                conciseness_score=float(p[1]),
                empathy_score=float(p[2])
            )
            for p in params
        ]


# ===================== 联合模型 =====================

class EmotionStyleModel(nn.Module):
    """
    联合模型：情感分类 + 风格预测
    
    将 EmotionClassifier 和 StylePredictor 组合为端到端模型
    """
    
    def __init__(self, emotion_classifier, style_predictor):
        super().__init__()
        self.emotion_classifier = emotion_classifier
        self.style_predictor = style_predictor
    
    def forward(self, video_feat=None, text_feat=None,
                context_emotions=None, context_mask=None,
                context_video_feat=None, context_text_feat=None,
                context_same_speaker=None, context_turn_distance=None,
                context_relation_ids=None,
                speaker_memory_video_feat=None,
                speaker_memory_text_feat=None,
                speaker_memory_mask=None):
        """
        端到端前向传播
        
        Returns:
            dict with emotion classification + style prediction results
        """
        # 1. 情感分类
        emotion_output = self.emotion_classifier(
            video_feat=video_feat, 
            text_feat=text_feat,
            context_video_feat=context_video_feat,
            context_text_feat=context_text_feat,
            context_mask=context_mask,
            context_same_speaker=context_same_speaker,
            context_turn_distance=context_turn_distance,
            context_relation_ids=context_relation_ids,
            speaker_memory_video_feat=speaker_memory_video_feat,
            speaker_memory_text_feat=speaker_memory_text_feat,
            speaker_memory_mask=speaker_memory_mask,
            return_features=True,
            return_attention=True
        )
        
        # 2. 风格预测（使用情感概率 + 融合特征）
        style_output = self.style_predictor(
            emotion_probs=emotion_output['probs'],
            context_emotions=context_emotions,
            context_mask=context_mask,
            emotion_features=emotion_output.get('features')
        )
        
        return {
            **emotion_output,
            **style_output,
        }


# ===================== 测试代码 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("可学习风格预测网络 - 结构验证")
    print("=" * 60)
    
    B = 4           # batch size
    T = 5           # 上下文长度（前5轮）
    num_emotions = 7
    
    # 模拟输入
    emotion_probs = F.softmax(torch.randn(B, num_emotions), dim=-1)
    context_emotions = F.softmax(torch.randn(B, T, num_emotions), dim=-1)
    context_mask = torch.ones(B, T)
    context_mask[:, -2:] = 0  # 最后两轮是padding
    
    # 1. 基础版（仅情感概率输入）
    print("\n--- 基础版 StylePredictor ---")
    predictor = StylePredictor(use_emotion_features=False)
    output = predictor(emotion_probs, context_emotions, context_mask)
    print(f"  style_params: {output['style_params'].shape}")      # [B, 3]
    print(f"  context_attn: {output['context_attention'].shape}")  # [B, T]
    print(f"  样本0的风格参数: {output['style_params'][0].tolist()}")
    
    # 解释性输出
    style_list = predictor.predict_style(emotion_probs, context_emotions)
    for i, sp in enumerate(style_list):
        print(f"  样本{i}: {sp.get_description()} | scores={sp.to_discrete()}")
    
    # 2. 增强版（使用融合特征）
    print("\n--- 增强版 StylePredictor (with features) ---")
    predictor2 = StylePredictor(use_emotion_features=True, feature_dim=256)
    emotion_features = torch.randn(B, 256)
    output2 = predictor2(emotion_probs, context_emotions, context_mask, emotion_features)
    print(f"  style_params: {output2['style_params'].shape}")
    
    # 3. 无上下文
    print("\n--- 无上下文 ---")
    output3 = predictor(emotion_probs)
    print(f"  style_params: {output3['style_params'].shape}")
    print(f"  context_attn: {output3['context_attention']}")  # None
    
    # 4. 训练标签生成
    print("\n--- 训练标签生成 ---")
    labels = torch.tensor([0, 3, 4, 6])  # neutral, sadness, joy, anger
    targets = generate_style_labels(labels)
    print(f"  targets shape: {targets.shape}")
    for i, name in enumerate(['neutral', 'sadness', 'joy', 'anger']):
        print(f"  {name:10s}: tone={targets[i,0]:.1f} concise={targets[i,1]:.1f} empathy={targets[i,2]:.1f}")
    
    # 5. 参数量
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"\n  总参数量: {total_params:,}")
    
    print("\n✓ 风格预测网络验证通过!")
