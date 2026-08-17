"""
多模态情感融合模型
实现三种融合策略：Concat / Cross-Modal Attention / Dynamic Gated Fusion
用于消融实验对比

Author: 陈裕瀚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


# ===================== Focal Loss (处理类别不平衡) =====================

class FocalLoss(nn.Module):
    """
    Focal Loss: 缓解MELD数据集中neutral类过多导致的类别不平衡问题
    
    参考: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha: 每个类别的权重，可根据MELD类别分布设置
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] 未经softmax的分类得分
            targets: [B] 类别标签 (0-6)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = softmax后的正确类概率
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ===================== 融合策略 =====================

class ConcatFusion(nn.Module):
    """
    基线融合方式：简单拼接 + 线性投影
    
    fused = ReLU(Linear(cat(video_feat, text_feat)))
    """
    
    def __init__(self, video_dim: int, text_dim: int, output_dim: int):
        super().__init__()
        # 输入归一化：原始Qwen特征需要归一化后再投影
        self.video_norm = nn.LayerNorm(video_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.projection = nn.Sequential(
            nn.Linear(video_dim + text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor, 
                return_weights: bool = False) -> torch.Tensor:
        """
        Args:
            video_feat: [B, D_v]
            text_feat:  [B, D_t]
        Returns:
            fused: [B, output_dim]
        """
        video_feat = self.video_norm(video_feat)
        text_feat = self.text_norm(text_feat)
        fused = self.projection(torch.cat([video_feat, text_feat], dim=-1))
        if return_weights:
            return fused, None  # concat无attention权重
        return fused


class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合（核心创新）
    
    文本和视频特征互相作为Query和Key/Value，实现双向跨模态交互。
    设计灵感来自ViLBERT的co-attention，但简化为单层双向结构。
    
    text_attended = Attention(Q=text, K=video, V=video)
    video_attended = Attention(Q=video, K=text, V=text)
    fused = LayerNorm(FFN(cat(text_attended, video_attended)))
    """
    
    def __init__(self, video_dim: int, text_dim: int, output_dim: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = output_dim // num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # 输入归一化：原始Qwen特征需要归一化后再投影
        self.video_norm = nn.LayerNorm(video_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        # 将两种模态投影到相同维度
        self.video_proj = nn.Linear(video_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # 文本关注视频 (T→V)
        self.t2v_q = nn.Linear(output_dim, output_dim)
        self.t2v_k = nn.Linear(output_dim, output_dim)
        self.t2v_v = nn.Linear(output_dim, output_dim)
        
        # 视频关注文本 (V→T)
        self.v2t_q = nn.Linear(output_dim, output_dim)
        self.v2t_k = nn.Linear(output_dim, output_dim)
        self.v2t_v = nn.Linear(output_dim, output_dim)
        
        # 融合后的FFN
        self.ffn = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.output_dim = output_dim
    
    def _scaled_dot_product_attention(self, Q, K, V):
        """
        计算缩放点积注意力
        
        Args:
            Q: [B, H, 1, d_k]
            K: [B, H, 1, d_k]  
            V: [B, H, 1, d_k]
        Returns:
            output: [B, H, 1, d_k]
            attn_weights: [B, H, 1, 1]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor,
                return_weights: bool = False) -> torch.Tensor:
        """
        Args:
            video_feat: [B, D_v]
            text_feat:  [B, D_t]
            return_weights: 是否返回注意力权重（用于可解释性分析）
        Returns:
            fused: [B, output_dim]
            attn_weights: (optional) 注意力权重字典
        """
        B = video_feat.size(0)
        
        # 归一化 + 投影到相同维度
        v = self.video_proj(self.video_norm(video_feat))  # [B, D]
        t = self.text_proj(self.text_norm(text_feat))      # [B, D]
        
        # Reshape为多头: [B, D] -> [B, H, 1, d_k]
        def reshape_to_heads(x):
            return x.view(B, self.num_heads, 1, self.d_k)
        
        # 文本关注视频 (Text queries Video)
        Q_t = reshape_to_heads(self.t2v_q(t))
        K_v = reshape_to_heads(self.t2v_k(v))
        V_v = reshape_to_heads(self.t2v_v(v))
        text_attended, t2v_weights = self._scaled_dot_product_attention(Q_t, K_v, V_v)
        text_attended = text_attended.view(B, self.output_dim)
        text_attended = self.layer_norm1(text_attended + t)  # 残差连接
        
        # 视频关注文本 (Video queries Text)
        Q_v = reshape_to_heads(self.v2t_q(v))
        K_t = reshape_to_heads(self.v2t_k(t))
        V_t = reshape_to_heads(self.v2t_v(t))
        video_attended, v2t_weights = self._scaled_dot_product_attention(Q_v, K_t, V_t)
        video_attended = video_attended.view(B, self.output_dim)
        video_attended = self.layer_norm1(video_attended + v)  # 残差连接
        
        # 拼接 + FFN
        combined = torch.cat([text_attended, video_attended], dim=-1)
        fused = self.layer_norm2(self.ffn(combined))
        
        if return_weights:
            weights = {
                't2v': t2v_weights.squeeze(-1).squeeze(-1),  # [B, H]
                'v2t': v2t_weights.squeeze(-1).squeeze(-1),  # [B, H]
            }
            return fused, weights
        return fused


class DynamicGatedFusion(nn.Module):
    """
    动态门控融合
    
    自动学习每个样本中文本/视频模态的重要性权重。
    gate = sigmoid(W_g(cat(video, text)))
    fused = gate * video + (1 - gate) * text
    """
    
    def __init__(self, video_dim: int, text_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 输入归一化：原始Qwen特征需要归一化后再投影
        self.video_norm = nn.LayerNorm(video_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        # 将两种模态投影到相同维度
        self.video_proj = nn.Linear(video_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor,
                return_weights: bool = False) -> torch.Tensor:
        """
        Args:
            video_feat: [B, D_v]
            text_feat:  [B, D_t]
        Returns:
            fused: [B, output_dim]
        """
        v = self.video_proj(self.video_norm(video_feat))
        t = self.text_proj(self.text_norm(text_feat))
        
        gate = self.gate_net(torch.cat([v, t], dim=-1))  # [B, D]
        fused = gate * v + (1 - gate) * t
        fused = self.layer_norm(fused)
        
        if return_weights:
            # 返回门控值作为可解释性信息
            gate_mean = gate.mean(dim=-1)  # [B], 每个样本的平均门控值
            return fused, {'gate': gate_mean}
        return fused


class TextVideoProjectionFusion(nn.Module):
    """
    Project text/video features into one emotion space, then fuse them with a
    lightweight gate. The projected vectors are exposed for Step 9 diagnostics.
    """

    def __init__(self, video_dim: int, text_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.video_norm = nn.LayerNorm(video_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        video_feat: torch.Tensor,
        text_feat: torch.Tensor,
        return_weights: bool = False,
        return_projections: bool = False,
    ) -> torch.Tensor:
        video_proj = self.video_projection(self.video_norm(video_feat))
        text_proj = self.text_projection(self.text_norm(text_feat))
        gate = self.gate_net(torch.cat([video_proj, text_proj], dim=-1))
        fused = self.fusion_norm(gate * video_proj + (1.0 - gate) * text_proj)

        weights = None
        if return_weights:
            weights = {
                "gate": gate.mean(dim=-1),
                "projection_gate": gate.mean(dim=-1),
                "projection_video_norm": video_proj.norm(dim=-1),
                "projection_text_norm": text_proj.norm(dim=-1),
                "projection_cosine": F.cosine_similarity(text_proj, video_proj, dim=-1),
            }
        if return_projections:
            projections = {"video": video_proj, "text": text_proj}
            return fused, weights, projections
        if return_weights:
            return fused, weights
        return fused


class TextVideoDisagreementFusion(nn.Module):
    """
    Step 10 fusion: estimate unsupervised text-video disagreement from the
    single-modality emotion distributions, then use it to gate a robust path.
    """

    def __init__(
        self,
        video_dim: int,
        text_dim: int,
        output_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        disagreement_gate_min: float = 0.0,
        disagreement_gate_temperature: float = 1.0,
        disagreement_gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.disagreement_gate_min = float(disagreement_gate_min)
        self.disagreement_gate_temperature = max(float(disagreement_gate_temperature), 1e-6)
        self.disagreement_gate_bias_init = float(disagreement_gate_bias_init)
        self.video_norm = nn.LayerNorm(video_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.text_emotion_head = nn.Linear(output_dim, num_classes)
        self.video_emotion_head = nn.Linear(output_dim, num_classes)
        self.normal_gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.robust_path = nn.Sequential(
            nn.Linear(output_dim * 4 + 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.disagreement_gate_net = nn.Sequential(
            nn.Linear(output_dim * 2 + 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )
        self.fusion_norm = nn.LayerNorm(output_dim)

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return (p * ((p + eps).log() - (q + eps).log())).sum(dim=-1)

    def configure_gate_bias(self) -> None:
        final_linear = self.disagreement_gate_net[-1]
        if isinstance(final_linear, nn.Linear) and final_linear.bias is not None:
            nn.init.constant_(final_linear.bias, self.disagreement_gate_bias_init)

    def forward(
        self,
        video_feat: torch.Tensor,
        text_feat: torch.Tensor,
        return_weights: bool = False,
        return_projections: bool = False,
    ) -> torch.Tensor:
        video_proj = self.video_projection(self.video_norm(video_feat))
        text_proj = self.text_projection(self.text_norm(text_feat))
        text_logits = self.text_emotion_head(text_proj)
        video_logits = self.video_emotion_head(video_proj)
        text_probs = F.softmax(text_logits, dim=-1)
        video_probs = F.softmax(video_logits, dim=-1)

        mixture = 0.5 * (text_probs + video_probs)
        text_to_video_kl = self._kl_divergence(text_probs, video_probs)
        video_to_text_kl = self._kl_divergence(video_probs, text_probs)
        js_divergence = 0.5 * (
            self._kl_divergence(text_probs, mixture)
            + self._kl_divergence(video_probs, mixture)
        )
        disagreement = js_divergence.unsqueeze(-1)
        tv_abs_diff = (text_proj - video_proj).abs()
        tv_product = text_proj * video_proj
        modality_quality = torch.stack(
            [video_proj.norm(dim=-1), text_proj.norm(dim=-1)],
            dim=-1,
        )

        normal_gate = self.normal_gate_net(torch.cat([video_proj, text_proj], dim=-1))
        normal_path = normal_gate * video_proj + (1.0 - normal_gate) * text_proj
        robust_input = torch.cat(
            [text_proj, video_proj, tv_abs_diff, tv_product, disagreement, modality_quality],
            dim=-1,
        )
        robust_path = self.robust_path(robust_input)
        gate_input = torch.cat([text_proj, video_proj, disagreement, modality_quality], dim=-1)
        disagreement_gate_logit = self.disagreement_gate_net(gate_input)
        disagreement_gate = torch.sigmoid(disagreement_gate_logit / self.disagreement_gate_temperature)
        if self.disagreement_gate_min > 0.0:
            disagreement_gate = self.disagreement_gate_min + (1.0 - self.disagreement_gate_min) * disagreement_gate
        fused = self.fusion_norm(
            disagreement_gate * robust_path + (1.0 - disagreement_gate) * normal_path
        )

        weights = None
        if return_weights:
            weights = {
                "gate": normal_gate.mean(dim=-1),
                "projection_gate": normal_gate.mean(dim=-1),
                "projection_video_norm": video_proj.norm(dim=-1),
                "projection_text_norm": text_proj.norm(dim=-1),
                "projection_cosine": F.cosine_similarity(text_proj, video_proj, dim=-1),
                "disagreement_gate": disagreement_gate.squeeze(-1),
                "js_divergence": js_divergence,
                "kl_text_video": text_to_video_kl,
                "kl_video_text": video_to_text_kl,
            }
        projections = {
            "video": video_proj,
            "text": text_proj,
            "text_logits": text_logits,
            "video_logits": video_logits,
            "text_probs": text_probs,
            "video_probs": video_probs,
            "js_divergence": js_divergence,
            "kl_text_video": text_to_video_kl,
            "kl_video_text": video_to_text_kl,
            "disagreement_gate": disagreement_gate.squeeze(-1),
        }
        if return_projections:
            return fused, weights, projections
        if return_weights:
            return fused, weights
        return fused


class RelationAwareContextEncoder(nn.Module):
    """
    Encode previous K utterances with optional audited dialogue relation cues.

    Modes:
    - context: previous utterance features only
    - speaker_distance: add same-speaker and turn-distance embeddings
    - validated_relations: add Step 4 validated relation ids on top of
      speaker/distance cues; failed or uncertain weak relations remain unknown
    """

    def __init__(
        self,
        hidden_dim: int,
        max_distance: int = 20,
        use_relation_embeddings: bool = True,
        use_validated_relations: bool = False,
        num_relations: int = 4,
        dropout: float = 0.1,
        structure_gate_scale: float = 1.0,
        relation_dropout: float = 0.0,
        relation_embedding_init_std: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        self.use_relation_embeddings = use_relation_embeddings
        self.use_validated_relations = use_validated_relations
        self.num_relations = num_relations
        self.structure_gate_scale = structure_gate_scale

        if use_relation_embeddings:
            self.same_speaker_embedding = nn.Embedding(2, hidden_dim)
            self.distance_embedding = nn.Embedding(max_distance + 1, hidden_dim)
        if use_validated_relations:
            self.validated_relation_embedding = nn.Embedding(num_relations, hidden_dim)
        if use_relation_embeddings or use_validated_relations:
            self.relation_norm = nn.LayerNorm(hidden_dim)
            self.relation_dropout = nn.Dropout(relation_dropout)
            if relation_embedding_init_std is not None:
                embeddings = []
                if use_relation_embeddings:
                    embeddings.extend([self.same_speaker_embedding, self.distance_embedding])
                if use_validated_relations:
                    embeddings.append(self.validated_relation_embedding)
                for embedding in embeddings:
                    nn.init.normal_(embedding.weight, mean=0.0, std=relation_embedding_init_std)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        same_speaker: Optional[torch.Tensor] = None,
        turn_distances: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            current: [B, H] current utterance representation
            context: [B, T, H] previous utterance representations
            context_mask: [B, T], 1 for valid context rows
            same_speaker: [B, T], binary same-speaker flag
            turn_distances: [B, T], integer turn distances
            relation_ids: [B, T], Step 4 validated relation ids
        """
        if context is None or context_mask is None or context.size(1) == 0:
            empty = current.new_zeros(current.size(0), 0)
            stats = {
                "structure_attention": empty,
                "structure_gate": current.new_zeros(current.size(0)),
                "structure_valid_context": current.new_zeros(current.size(0)),
            }
            return current, stats

        mask = context_mask.float()
        enriched = context

        if self.use_relation_embeddings or self.use_validated_relations:
            relation = torch.zeros_like(context)

        if self.use_relation_embeddings:
            if same_speaker is None:
                same_speaker = torch.zeros_like(mask, dtype=torch.long)
            else:
                same_speaker = same_speaker.long().clamp(0, 1)

            if turn_distances is None:
                turn_distances = torch.zeros_like(same_speaker)
            else:
                turn_distances = turn_distances.long().clamp(0, self.max_distance)

            relation = relation + (
                self.same_speaker_embedding(same_speaker)
                + self.distance_embedding(turn_distances)
            )

        if self.use_validated_relations:
            if relation_ids is None:
                relation_ids = torch.zeros_like(mask, dtype=torch.long)
            else:
                relation_ids = relation_ids.long().clamp(0, self.num_relations - 1)
            relation = relation + self.validated_relation_embedding(relation_ids)

        if self.use_relation_embeddings or self.use_validated_relations:
            relation = self.relation_dropout(relation)
            enriched = self.relation_norm(context + relation)

        query = self.query_proj(current).unsqueeze(1)  # [B, 1, H]
        keys = self.key_proj(enriched)                 # [B, T, H]
        values = self.value_proj(enriched)             # [B, T, H]

        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(mask <= 0, -1e4)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        attention = attention * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context_repr = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        combined = torch.cat([current, context_repr], dim=-1)
        update = self.context_update(combined)
        gate = self.context_gate(combined) * self.structure_gate_scale
        fused = self.layer_norm((1.0 - gate) * current + gate * update)

        stats = {
            "structure_attention": attention,
            "structure_gate": gate.mean(dim=-1),
            "structure_valid_context": mask.sum(dim=-1),
        }
        return fused, stats


class SpeakerPrototypeMemoryEncoder(nn.Module):
    """Attend over fixed offline speaker prototype slots."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, memory_gate_scale: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_gate_scale = memory_gate_scale
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.memory_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if memory is None or memory_mask is None or memory.size(1) == 0:
            empty = current.new_zeros(current.size(0), 0)
            return current, {
                "memory_attention": empty,
                "memory_gate": current.new_zeros(current.size(0)),
                "memory_valid_slots": current.new_zeros(current.size(0)),
            }

        mask = memory_mask.float()
        query = self.query_proj(current).unsqueeze(1)
        keys = self.key_proj(memory)
        values = self.value_proj(memory)

        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(mask <= 0, -1e4)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        attention = attention * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        memory_repr = torch.bmm(attention.unsqueeze(1), values).squeeze(1)
        combined = torch.cat([current, memory_repr], dim=-1)
        update = self.memory_update(combined)
        gate = self.memory_gate(combined) * self.memory_gate_scale
        fused = self.layer_norm((1.0 - gate) * current + gate * update)

        return fused, {
            "memory_attention": attention,
            "memory_gate": gate.mean(dim=-1),
            "memory_valid_slots": mask.sum(dim=-1),
        }


# ===================== 完整分类模型 =====================

class EmotionClassifier(nn.Module):
    """
    完整的多模态情感分类器
    
    支持三种模式：
    1. 多模态融合（video + text）
    2. 纯文本
    3. 纯视频
    
    支持三种融合策略：concat / attention / gated
    """
    
    FUSION_TYPES = {
        'concat': ConcatFusion,
        'attention': CrossModalAttention,
        'gated': DynamicGatedFusion,
        'tv_projection': TextVideoProjectionFusion,
        'tv_disagreement': TextVideoDisagreementFusion,
    }
    
    def __init__(self, video_dim: int = 3584, text_dim: int = 3584,
                 hidden_dim: int = 256, num_classes: int = 7,
                 fusion_type: str = 'attention', dropout: float = 0.3,
                 num_heads: int = 4, structure_mode: str = 'none',
                 max_context_len: int = 5, max_distance: int = 20,
                 num_relations: int = 4, structure_gate_scale: float = 1.0,
                 relation_dropout: float = 0.0,
                 relation_embedding_init_std: Optional[float] = None,
                 speaker_memory_mode: str = 'none',
                 speaker_memory_slots: int = 4,
                 memory_gate_scale: float = 1.0,
                 disagreement_gate_min: float = 0.0,
                 disagreement_gate_temperature: float = 1.0,
                 disagreement_gate_bias_init: float = 0.0):
        """
        Args:
            video_dim: 视频特征维度 (Qwen2.5-Omni hidden size)
            text_dim: 文本特征维度
            hidden_dim: 融合后的隐层维度
            num_classes: 分类数（MELD=7）
            fusion_type: 融合策略 ('concat', 'attention', 'gated')
            dropout: Dropout比率
            num_heads: 注意力头数（仅attention模式）
            structure_mode: none/context/speaker_distance/validated_relations
            max_context_len: 最大结构上下文长度
            max_distance: turn distance embedding 的最大距离
            num_relations: Step 4 relation id 词表大小
            structure_gate_scale: 结构上下文 gate 缩放，1.0 保持原行为
            relation_dropout: relation/speaker/distance embedding dropout
            relation_embedding_init_std: relation embedding 小方差初始化；None 保持默认初始化
            speaker_memory_mode: none/prototype，是否启用固定 speaker prototype memory
            speaker_memory_slots: 每个 speaker 的固定 prototype slot 数
            memory_gate_scale: speaker memory gate 缩放
            disagreement_gate_min: Step 10.5 disagreement gate 下限，0 保持原行为
            disagreement_gate_temperature: Step 10.5 gate sigmoid 温度
            disagreement_gate_bias_init: Step 10.5 gate 最后一层 bias 初始化
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.structure_mode = structure_mode
        self.max_context_len = max_context_len
        self.speaker_memory_mode = speaker_memory_mode
        self.speaker_memory_slots = speaker_memory_slots
        
        # 融合模块
        if fusion_type in self.FUSION_TYPES:
            fusion_cls = self.FUSION_TYPES[fusion_type]
            if fusion_type == 'attention':
                self.fusion = fusion_cls(video_dim, text_dim, hidden_dim, 
                                        num_heads=num_heads, dropout=dropout * 0.5)
            elif fusion_type == 'tv_projection':
                self.fusion = fusion_cls(video_dim, text_dim, hidden_dim, dropout=dropout * 0.5)
                self.text_emotion_head = nn.Linear(hidden_dim, num_classes)
                self.video_emotion_head = nn.Linear(hidden_dim, num_classes)
            elif fusion_type == 'tv_disagreement':
                self.fusion = fusion_cls(
                    video_dim,
                    text_dim,
                    hidden_dim,
                    num_classes,
                    dropout=dropout * 0.5,
                    disagreement_gate_min=disagreement_gate_min,
                    disagreement_gate_temperature=disagreement_gate_temperature,
                    disagreement_gate_bias_init=disagreement_gate_bias_init,
                )
            else:
                self.fusion = fusion_cls(video_dim, text_dim, hidden_dim)
        elif fusion_type == 'text_only':
            self.text_proj = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        elif fusion_type == 'video_only':
            self.video_proj = nn.Sequential(
                nn.LayerNorm(video_dim),
                nn.Linear(video_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        if structure_mode not in ('none', 'context', 'speaker_distance', 'validated_relations'):
            raise ValueError(f"Unknown structure_mode: {structure_mode}")

        self.use_structure_context = structure_mode != 'none'
        if self.use_structure_context:
            self.context_encoder = RelationAwareContextEncoder(
                hidden_dim=hidden_dim,
                max_distance=max_distance,
                use_relation_embeddings=(structure_mode in ('speaker_distance', 'validated_relations')),
                use_validated_relations=(structure_mode == 'validated_relations'),
                num_relations=num_relations,
                dropout=dropout,
                structure_gate_scale=structure_gate_scale,
                relation_dropout=relation_dropout,
                relation_embedding_init_std=relation_embedding_init_std,
            )

        if speaker_memory_mode not in ('none', 'prototype'):
            raise ValueError(f"Unknown speaker_memory_mode: {speaker_memory_mode}")
        self.use_speaker_memory = speaker_memory_mode == 'prototype'
        if self.use_speaker_memory:
            self.speaker_memory_encoder = SpeakerPrototypeMemoryEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                memory_gate_scale=memory_gate_scale,
            )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        if fusion_type == 'tv_disagreement' and hasattr(self.fusion, 'configure_gate_bias'):
            self.fusion.configure_gate_bias()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _encode_context_features(
        self,
        context_video_feat: Optional[torch.Tensor],
        context_text_feat: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Encode padded context utterance features with the same base encoder."""
        source = context_text_feat if context_text_feat is not None else context_video_feat
        if source is None:
            return None

        B, T = source.shape[:2]
        if T == 0:
            return source.new_zeros(B, 0, self.hidden_dim)

        if self.fusion_type in ('concat', 'attention', 'gated', 'tv_projection', 'tv_disagreement'):
            assert context_video_feat is not None and context_text_feat is not None, \
                f"{self.fusion_type} context encoding requires both video and text features"
            flat_v = context_video_feat.reshape(B * T, -1)
            flat_t = context_text_feat.reshape(B * T, -1)
            encoded = self.fusion(flat_v, flat_t)
        elif self.fusion_type == 'text_only':
            assert context_text_feat is not None, "text_only context encoding requires text features"
            encoded = self.text_proj(context_text_feat.reshape(B * T, -1))
        elif self.fusion_type == 'video_only':
            assert context_video_feat is not None, "video_only context encoding requires video features"
            encoded = self.video_proj(context_video_feat.reshape(B * T, -1))
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return encoded.view(B, T, self.hidden_dim)

    def forward(self, video_feat: Optional[torch.Tensor] = None,
                text_feat: Optional[torch.Tensor] = None,
                context_video_feat: Optional[torch.Tensor] = None,
                context_text_feat: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                context_same_speaker: Optional[torch.Tensor] = None,
                context_turn_distance: Optional[torch.Tensor] = None,
                context_relation_ids: Optional[torch.Tensor] = None,
                speaker_memory_video_feat: Optional[torch.Tensor] = None,
                speaker_memory_text_feat: Optional[torch.Tensor] = None,
                speaker_memory_mask: Optional[torch.Tensor] = None,
                return_features: bool = False,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_feat: [B, D_v] 视频特征
            text_feat:  [B, D_t] 文本特征
            return_features: 是否返回融合后的特征（用于风格预测）
            return_attention: 是否返回注意力权重（用于可解释性）
            context_*: 前 K 轮上下文特征和稳定结构关系字段
        
        Returns:
            dict with keys:
                'logits': [B, num_classes]
                'probs': [B, num_classes]
                'features': [B, hidden_dim] (if return_features)
                'attention_weights': dict (if return_attention)
        """
        # 获取融合特征
        attn_weights = None
        
        projection_features = None

        if self.fusion_type in ('concat', 'attention', 'gated'):
            assert video_feat is not None and text_feat is not None, \
                f"{self.fusion_type} fusion requires both video and text features"
            if return_attention:
                fused, attn_weights = self.fusion(video_feat, text_feat, return_weights=True)
            else:
                fused = self.fusion(video_feat, text_feat)
        elif self.fusion_type in ('tv_projection', 'tv_disagreement'):
            assert video_feat is not None and text_feat is not None, \
                f"{self.fusion_type} fusion requires both video and text features"
            fused, attn_weights, projection_features = self.fusion(
                video_feat,
                text_feat,
                return_weights=return_attention,
                return_projections=True,
            )
        elif self.fusion_type == 'text_only':
            assert text_feat is not None, "text_only mode requires text features"
            fused = self.text_proj(text_feat)
        elif self.fusion_type == 'video_only':
            assert video_feat is not None, "video_only mode requires video features"
            fused = self.video_proj(video_feat)

        if self.use_structure_context and context_mask is not None:
            context_hidden = self._encode_context_features(context_video_feat, context_text_feat)
            if context_hidden is not None:
                fused, context_weights = self.context_encoder(
                    current=fused,
                    context=context_hidden,
                    context_mask=context_mask,
                    same_speaker=context_same_speaker,
                    turn_distances=context_turn_distance,
                    relation_ids=context_relation_ids,
                )
                if attn_weights is None:
                    attn_weights = {}
                attn_weights.update(context_weights)

        if self.use_speaker_memory and speaker_memory_mask is not None:
            memory_hidden = self._encode_context_features(speaker_memory_video_feat, speaker_memory_text_feat)
            if memory_hidden is not None:
                fused, memory_weights = self.speaker_memory_encoder(
                    current=fused,
                    memory=memory_hidden,
                    memory_mask=speaker_memory_mask,
                )
                if attn_weights is None:
                    attn_weights = {}
                attn_weights.update(memory_weights)
        
        # 分类
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
        }
        if projection_features is not None:
            if "text_logits" in projection_features and "video_logits" in projection_features:
                text_logits = projection_features["text_logits"]
                video_logits = projection_features["video_logits"]
            else:
                text_logits = self.text_emotion_head(projection_features["text"])
                video_logits = self.video_emotion_head(projection_features["video"])
            output["text_logits"] = text_logits
            output["video_logits"] = video_logits
            output["text_probs"] = projection_features.get("text_probs", F.softmax(text_logits, dim=-1))
            output["video_probs"] = projection_features.get("video_probs", F.softmax(video_logits, dim=-1))
            for key in ("js_divergence", "kl_text_video", "kl_video_text", "disagreement_gate"):
                if key in projection_features:
                    output[key] = projection_features[key]
            if return_features:
                output["text_projection"] = projection_features["text"]
                output["video_projection"] = projection_features["video"]
        
        if return_features:
            output['features'] = fused
        if return_attention and attn_weights is not None:
            output['attention_weights'] = attn_weights
        
        return output
    
    def get_param_count(self) -> Dict[str, int]:
        """获取各模块的参数量"""
        counts = {}
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        counts['total'] = total
        counts['trainable'] = trainable
        
        if hasattr(self, 'fusion'):
            counts['fusion'] = sum(p.numel() for p in self.fusion.parameters())
        if hasattr(self, 'context_encoder'):
            counts['context_encoder'] = sum(p.numel() for p in self.context_encoder.parameters())
        if hasattr(self, 'speaker_memory_encoder'):
            counts['speaker_memory_encoder'] = sum(p.numel() for p in self.speaker_memory_encoder.parameters())
        counts['classifier'] = sum(p.numel() for p in self.classifier.parameters())
        
        return counts


# ===================== 工具函数 =====================

def get_class_weights(
    dataset: str = 'meld',
    label_counts: Optional[Dict[int, int]] = None,
    mode: str = 'inverse',
) -> torch.Tensor:
    """
    计算不同数据集的类别权重（用于Focal Loss或加权交叉熵）
    
    Returns:
        weights: 类别权重张量
    """
    if label_counts is None:
        if dataset == 'meld':
            # MELD train set 默认分布
            label_counts = {
                0: 4710,  # neutral
                1: 1205,  # surprise
                2: 268,   # fear
                3: 683,   # sadness
                4: 1743,  # joy
                5: 271,   # disgust
                6: 1109,  # anger
            }
        elif dataset == 'chsims':
            # 基于阈值0.3的估计分布，若提供确切counts会更好
            label_counts = {
                0: 700,   # negative (假设)
                1: 900,   # neutral
                2: 681    # positive
            }
        elif dataset == 'mosei':
            label_counts = {i: 1000 for i in range(6)} # Placeholder
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    if mode not in {'inverse', 'sqrt_inverse'}:
        raise ValueError(f"Unsupported class weight mode: {mode}")

    total = sum(label_counts.values())
    num_classes = len(label_counts)
    
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        # 逆频率权重，然后归一化
        if count == 0: count = 1
        if mode == 'inverse':
            w = total / (num_classes * count)
        else:
            w = math.sqrt(total / count)
        weights.append(w)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    # 归一化使得均值为1
    weights = weights / weights.mean()
    
    return weights


# ===================== 测试代码 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("多模态情感融合模型 - 结构验证")
    print("=" * 60)
    
    B = 8  # batch size
    D_v = 3584  # Qwen2.5-Omni hidden dim
    D_t = 3584
    
    # 生成假数据
    video_feat = torch.randn(B, D_v)
    text_feat = torch.randn(B, D_t)
    labels = torch.randint(0, 7, (B,))
    
    for fusion_type in ['concat', 'attention', 'gated', 'tv_projection', 'tv_disagreement', 'text_only', 'video_only']:
        print(f"\n--- {fusion_type.upper()} ---")
        model = EmotionClassifier(
            video_dim=D_v, text_dim=D_t,
            hidden_dim=256, num_classes=7,
            fusion_type=fusion_type
        )
        
        # 前向传播
        if fusion_type == 'text_only':
            output = model(text_feat=text_feat, return_features=True, return_attention=True)
        elif fusion_type == 'video_only':
            output = model(video_feat=video_feat, return_features=True, return_attention=True)
        else:
            output = model(video_feat=video_feat, text_feat=text_feat, 
                         return_features=True, return_attention=True)
        
        print(f"  logits shape: {output['logits'].shape}")
        print(f"  probs shape:  {output['probs'].shape}")
        print(f"  features:     {output['features'].shape}")
        
        if 'attention_weights' in output and output['attention_weights'] is not None:
            for k, v in output['attention_weights'].items():
                print(f"  attn [{k}]:    {v.shape}")
        
        # 参数量
        params = model.get_param_count()
        print(f"  params: total={params['total']:,}, trainable={params['trainable']:,}")
        
        # 测试loss
        criterion = FocalLoss(alpha=get_class_weights('meld'), gamma=2.0)
        loss = criterion(output['logits'], labels)
        print(f"  focal loss:   {loss.item():.4f}")
    
    print("\n✓ 所有融合模型验证通过!")
