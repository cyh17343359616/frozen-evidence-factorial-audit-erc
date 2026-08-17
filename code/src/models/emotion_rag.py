"""
Emotion-RAG: 检索增强的上下文感知情感识别模型

将 RAG（检索增强生成）范式迁移到多模态情感识别：
1. 构建情感记忆库（Emotion Memory Bank），用 FAISS 索引训练集融合特征
2. 推理时检索 Top-K 最相似样本，将其特征和标签作为上下文注入
3. 通过 Cross-Attention 融合当前特征和检索上下文，增强情感预测

支持三种检索策略：
- global: 全训练集检索
- speaker: 仅检索同说话人样本（风格自适应）
- dialogue: 仅检索同对话前几句（上下文感知）

Author: 陈裕瀚
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple


# ===================== 情感记忆库 =====================

class EmotionMemoryBank:
    """
    基于 FAISS 的情感记忆库

    存储训练集中所有样本的融合特征，支持按语义相似度检索 Top-K 样本。
    可选按 speaker_id 或 dialogue_id 过滤检索范围。
    如果本地环境未安装 FAISS，则自动退回到 NumPy 内积检索，用于小规模复核。
    """
    
    def __init__(self, feature_dim: int, use_gpu: bool = False):
        """
        Args:
            feature_dim: 融合特征维度（hidden_dim）
            use_gpu: 是否使用 GPU 加速检索
        """
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        self.faiss = None
        self.index = None

        try:
            import faiss

            self.faiss = faiss
            self.index = faiss.IndexFlatIP(feature_dim)  # 内积检索

            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except ImportError:
            self.faiss = None
            self.index = None
        
        # 元信息
        self.features = None      # [N, D] 所有特征
        self.labels = None        # [N] 对应的情感标签
        self.speaker_ids = None   # [N] 说话人 ID（可选）
        self.dialogue_ids = None  # [N] 对话 ID（可选）
        self.utterance_ids = None # [N] 句子序号（可选）
        
        # 按 speaker/dialogue 分组的索引映射
        self._speaker_indices = {}   # speaker_id -> [indices]
        self._dialogue_indices = {}  # dialogue_id -> [indices]

    @staticmethod
    def _normalize_rows(array: np.ndarray) -> np.ndarray:
        """L2 normalize rows without requiring FAISS."""
        array = array.astype(np.float32, copy=True)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return array / norms
    
    def build(self, features: np.ndarray, labels: np.ndarray,
              speaker_ids: Optional[np.ndarray] = None,
              dialogue_ids: Optional[np.ndarray] = None,
              utterance_ids: Optional[np.ndarray] = None):
        """
        构建记忆库
        
        Args:
            features: [N, D] L2 归一化后的融合特征
            labels: [N] 情感标签
            speaker_ids: [N] 说话人 ID（可选）
            dialogue_ids: [N] 对话 ID（可选）
            utterance_ids: [N] 句子序号（可选）
        """
        # L2 归一化（确保内积等价于余弦相似度）
        if self.faiss is not None:
            features = features.astype(np.float32, copy=True)
            self.faiss.normalize_L2(features)
        else:
            features = self._normalize_rows(features)

        self.features = features.copy()
        self.labels = labels.copy()
        self.speaker_ids = speaker_ids.copy() if speaker_ids is not None else None
        self.dialogue_ids = dialogue_ids.copy() if dialogue_ids is not None else None
        self.utterance_ids = utterance_ids.copy() if utterance_ids is not None else None
        
        # 添加到 FAISS 索引
        if self.index is not None:
            self.index.reset()
            self.index.add(features)
        
        # 构建分组索引
        if speaker_ids is not None:
            self._speaker_indices = {}
            for i, sid in enumerate(speaker_ids):
                sid_key = str(sid)
                if sid_key not in self._speaker_indices:
                    self._speaker_indices[sid_key] = []
                self._speaker_indices[sid_key].append(i)
        
        if dialogue_ids is not None:
            self._dialogue_indices = {}
            for i, did in enumerate(dialogue_ids):
                did_key = str(did)
                if did_key not in self._dialogue_indices:
                    self._dialogue_indices[did_key] = []
                self._dialogue_indices[did_key].append(i)
        
        backend = "FAISS" if self.index is not None else "NumPy"
        print(f"  ✓ EmotionMemoryBank 构建完成: {len(features)} 样本, dim={self.feature_dim}, backend={backend}")
        if speaker_ids is not None:
            print(f"    说话人数: {len(self._speaker_indices)}")
        if dialogue_ids is not None:
            print(f"    对话数: {len(self._dialogue_indices)}")
    
    def retrieve_global(self, query: np.ndarray, top_k: int = 5,
                        exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        全局检索 Top-K 最相似样本
        
        Args:
            query: [D] 或 [1, D] 查询特征
            top_k: 检索数量
            exclude_idx: 排除的样本索引（避免检索到自身）
            
        Returns:
            features: [K, D] 检索到的特征
            labels: [K] 检索到的标签
            scores: [K] 相似度分数
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self.faiss is not None and self.index is not None:
            query = query.astype(np.float32, copy=True)
            self.faiss.normalize_L2(query)

            # 多检索几个以防需要排除自身
            search_k = top_k + 1 if exclude_idx is not None else top_k
            scores, indices = self.index.search(query, min(search_k, len(self.features)))
            scores = scores[0]
            indices = indices[0]
        else:
            query = self._normalize_rows(query)
            similarities = (query @ self.features.T).squeeze(0)
            if exclude_idx is not None and 0 <= exclude_idx < len(similarities):
                similarities[exclude_idx] = -np.inf
            search_k = min(top_k, len(similarities))
            indices = np.argsort(similarities)[::-1][:search_k]
            scores = similarities[indices]
        
        # 排除自身
        if exclude_idx is not None:
            mask = indices != exclude_idx
            indices = indices[mask][:top_k]
            scores = scores[mask][:top_k]
        else:
            indices = indices[:top_k]
            scores = scores[:top_k]
        
        if len(indices) < top_k:
            return self._pad_result(indices, scores, top_k)

        return self.features[indices], self.labels[indices], scores.astype(np.float32)
    
    def retrieve_by_speaker(self, query: np.ndarray, speaker_id,
                            top_k: int = 5, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """仅从同说话人的样本中检索"""
        sid_key = str(speaker_id)
        if sid_key not in self._speaker_indices or len(self._speaker_indices[sid_key]) == 0:
            return self._empty_result(top_k)
        
        candidate_indices = np.array(self._speaker_indices[sid_key])
        return self._retrieve_from_subset(query, candidate_indices, top_k, exclude_idx)
    
    def retrieve_by_dialogue(self, query: np.ndarray, dialogue_id,
                             utterance_id=None, top_k: int = 5,
                             exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """仅从同对话的前几句中检索"""
        did_key = str(dialogue_id)
        if did_key not in self._dialogue_indices or len(self._dialogue_indices[did_key]) == 0:
            return self._empty_result(top_k)
        
        candidate_indices = np.array(self._dialogue_indices[did_key])
        
        # 如果提供了 utterance_id，只检索该对话中排在当前句之前的样本
        if utterance_id is not None and self.utterance_ids is not None:
            prev_mask = self.utterance_ids[candidate_indices] < utterance_id
            candidate_indices = candidate_indices[prev_mask]
            if len(candidate_indices) == 0:
                return self._empty_result(top_k)
        
        return self._retrieve_from_subset(query, candidate_indices, top_k, exclude_idx)
    
    def _retrieve_from_subset(self, query: np.ndarray, candidate_indices: np.ndarray,
                              top_k: int, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从指定的子集中检索 Top-K"""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = self._normalize_rows(query)
        
        # 排除自身
        if exclude_idx is not None:
            candidate_indices = candidate_indices[candidate_indices != exclude_idx]
        
        if len(candidate_indices) == 0:
            return self._empty_result(top_k)
        
        # 从子集中计算相似度
        subset_features = self.features[candidate_indices]
        similarities = (query @ subset_features.T).squeeze(0)  # [num_candidates]
        
        actual_k = min(top_k, len(similarities))
        top_k_local = np.argsort(similarities)[::-1][:actual_k]
        
        indices = candidate_indices[top_k_local]
        scores = similarities[top_k_local]
        
        # Pad if needed
        if actual_k < top_k:
            pad_feats = np.zeros((top_k - actual_k, self.feature_dim), dtype=np.float32)
            pad_labels = np.full(top_k - actual_k, -1, dtype=np.int64)
            pad_scores = np.zeros(top_k - actual_k, dtype=np.float32)
            return (
                np.concatenate([self.features[indices], pad_feats]),
                np.concatenate([self.labels[indices], pad_labels]),
                np.concatenate([scores, pad_scores])
            )
        
        return self.features[indices], self.labels[indices], scores.astype(np.float32)

    def _pad_result(self, indices: np.ndarray, scores: np.ndarray, top_k: int):
        """Pad an indexed retrieval result to top_k."""
        actual_k = len(indices)
        if actual_k == 0:
            return self._empty_result(top_k)
        pad = top_k - actual_k
        return (
            np.concatenate([
                self.features[indices],
                np.zeros((pad, self.feature_dim), dtype=np.float32),
            ]),
            np.concatenate([
                self.labels[indices],
                np.full(pad, -1, dtype=np.int64),
            ]),
            np.concatenate([
                scores.astype(np.float32),
                np.zeros(pad, dtype=np.float32),
            ]),
        )
    
    def _empty_result(self, top_k: int):
        """没有候选时返回全零结果"""
        return (
            np.zeros((top_k, self.feature_dim), dtype=np.float32),
            np.full(top_k, -1, dtype=np.int64),
            np.zeros(top_k, dtype=np.float32)
        )


# ===================== RAG 增强模块 =====================

class RAGAugmentation(nn.Module):
    """
    RAG 检索增强模块
    
    将当前样本特征与检索到的 Top-K 特征通过 Cross-Attention 融合。
    检索标签通过 Embedding 编码后也参与融合。
    
    h_augmented = h_current + α · CrossAttn(h_current, [H_retrieved; LabelEmbed])
    """
    
    def __init__(self, hidden_dim: int, num_classes: int, num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            hidden_dim: 融合特征维度
            num_classes: 情感类别数
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 检索标签 Embedding（+1 for padding label=-1）
        self.label_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
        
        # Cross-Attention: Query=当前样本, Key/Value=检索样本
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合 gate：控制 RAG 增强的程度
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_current: torch.Tensor,
                retrieved_features: torch.Tensor,
                retrieved_labels: torch.Tensor,
                retrieved_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h_current: [B, D] 当前样本的融合特征
            retrieved_features: [B, K, D] 检索到的特征
            retrieved_labels: [B, K] 检索到的标签 (可能包含 -1 = padding)
            retrieved_scores: [B, K] 相似度分数
            
        Returns:
            h_augmented: [B, D] 增强后的特征
            info: dict with attention weights and gate values
        """
        B, K, D = retrieved_features.shape
        
        # 标签 Embedding（将 -1 映射到 padding idx）
        labels_for_embed = retrieved_labels.clone()
        labels_for_embed[labels_for_embed < 0] = self.label_embed.padding_idx
        label_embeds = self.label_embed(labels_for_embed)  # [B, K, D]
        
        # 检索特征 + 标签 embedding 相加
        kv_features = retrieved_features + label_embeds  # [B, K, D]
        
        # padding mask: retrieved_labels == -1 的位置是 padding
        key_padding_mask = (retrieved_labels < 0)  # [B, K], True = mask
        
        # 检测全 mask 的样本（无有效候选 → 跳过 RAG）
        all_masked = key_padding_mask.all(dim=1)  # [B], True = 该样本无有效检索
        
        # 如果整个 batch 都是全 mask，直接返回原特征
        if all_masked.all():
            info = {
                'rag_gate': torch.zeros(B, device=h_current.device),
                'attn_weights': None,
            }
            return self.norm(h_current), info
        
        # 对全 mask 的样本，临时设一个位置为 False，防止 softmax NaN
        # （后续会用 gate=0 抵消这些样本的 RAG 影响）
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked, 0] = False
        
        # Cross-Attention: Query=[B,1,D], Key/Value=[B,K,D]
        query = h_current.unsqueeze(1)  # [B, 1, D]
        
        attn_output, attn_weights = self.cross_attn(
            query, kv_features, kv_features,
            key_padding_mask=safe_mask
        )
        attn_output = attn_output.squeeze(1)  # [B, D]
        
        # NaN 安全：将任何 NaN 替换为 0
        attn_output = torch.nan_to_num(attn_output, nan=0.0)
        
        # 门控残差连接
        gate_input = torch.cat([h_current, attn_output], dim=-1)
        alpha = self.gate(gate_input)  # [B, 1]
        
        # 对全 mask 的样本，强制 gate=0（不使用 RAG）
        alpha = alpha * (~all_masked).float().unsqueeze(-1)
        
        h_augmented = h_current + alpha * self.dropout(attn_output)
        h_augmented = self.norm(h_augmented)
        
        info = {
            'rag_gate': alpha.squeeze(-1).detach(),       # [B]
            'attn_weights': attn_weights.squeeze(1).detach() if attn_weights is not None else None,  # [B, K]
        }
        
        return h_augmented, info


# ===================== 完整 RAG 分类器 =====================

class EmotionRAGClassifier(nn.Module):
    """
    Emotion-RAG 完整分类器
    
    = Gated Fusion + RAG Augmentation + MLP Classifier
    
    在训练和推理时，RAG 模块利用记忆库中的检索结果增强当前样本的表示。
    """
    
    def __init__(self, video_dim: int = 3584, text_dim: int = 3584,
                 hidden_dim: int = 128, num_classes: int = 7,
                 dropout: float = 0.3, num_heads: int = 4,
                 top_k: int = 5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.top_k = top_k
        
        # 复用 Gated Fusion
        try:
            from .fusion_model import DynamicGatedFusion
        except ImportError:
            try:
                from models.fusion_model import DynamicGatedFusion
            except ImportError:
                from fusion_model import DynamicGatedFusion
        self.fusion = DynamicGatedFusion(video_dim, text_dim, hidden_dim)
        
        # RAG 增强模块
        self.rag = RAGAugmentation(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout * 0.5
        )
        
        # 分类头（与 EmotionClassifier 保持一致）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor,
                retrieved_features: Optional[torch.Tensor] = None,
                retrieved_labels: Optional[torch.Tensor] = None,
                retrieved_scores: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_feat: [B, D_v] 视频特征
            text_feat: [B, D_t] 文本特征
            retrieved_features: [B, K, D_hidden] 检索到的融合特征
            retrieved_labels: [B, K] 检索到的标签
            retrieved_scores: [B, K] 相似度分数
            return_features: 是否返回融合特征
            
        Returns:
            dict with 'logits', 'probs', optionally 'features', 'rag_info'
        """
        # Step 1: Gated Fusion
        fused = self.fusion(video_feat, text_feat)  # [B, hidden_dim]
        
        # Step 2: RAG Augmentation（如果提供了检索结果）
        rag_info = {}
        if retrieved_features is not None:
            fused, rag_info = self.rag(
                fused, retrieved_features, retrieved_labels, retrieved_scores
            )
        
        # Step 3: Classification
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        
        output = {'logits': logits, 'probs': probs}
        if return_features:
            output['features'] = fused
        if rag_info:
            output['rag_info'] = rag_info
        
        return output
    
    def get_param_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'rag': sum(p.numel() for p in self.rag.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }


# ===================== 自检 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("Emotion-RAG 模型结构验证")
    print("=" * 60)
    
    B = 8
    D_v = D_t = 3584
    hidden = 128
    K = 5
    C = 7
    
    # 1. 测试 RAG 增强模块
    print("\n--- RAGAugmentation ---")
    rag = RAGAugmentation(hidden, C)
    h = torch.randn(B, hidden)
    r_feat = torch.randn(B, K, hidden)
    r_labels = torch.randint(0, C, (B, K))
    r_labels[0, -1] = -1  # 测试 padding
    r_scores = torch.rand(B, K)
    
    h_aug, info = rag(h, r_feat, r_labels, r_scores)
    print(f"  input:  {h.shape}")
    print(f"  output: {h_aug.shape}")
    print(f"  gate:   {info['rag_gate'].shape}, mean={info['rag_gate'].mean():.4f}")
    
    # 2. 测试完整模型
    print("\n--- EmotionRAGClassifier ---")
    model = EmotionRAGClassifier(D_v, D_t, hidden, C, top_k=K)
    video = torch.randn(B, D_v)
    text = torch.randn(B, D_t)
    
    # 无 RAG
    out1 = model(video, text)
    print(f"  无RAG: logits={out1['logits'].shape}")
    
    # 有 RAG
    out2 = model(video, text, r_feat, r_labels, r_scores, return_features=True)
    print(f"  有RAG: logits={out2['logits'].shape}, features={out2['features'].shape}")
    print(f"  RAG gate mean: {out2['rag_info']['rag_gate'].mean():.4f}")
    
    # 参数量
    params = model.get_param_count()
    print(f"\n  参数量: total={params['total']:,}, trainable={params['trainable']:,}")
    print(f"    fusion:     {params['fusion']:,}")
    print(f"    rag:        {params['rag']:,}")
    print(f"    classifier: {params['classifier']:,}")
    
    # 3. 测试 EmotionMemoryBank
    print("\n--- EmotionMemoryBank ---")
    bank = EmotionMemoryBank(hidden)
    fake_feats = np.random.randn(100, hidden).astype(np.float32)
    fake_labels = np.random.randint(0, C, 100).astype(np.int64)
    fake_speakers = np.array(['Monica'] * 30 + ['Ross'] * 30 + ['Joey'] * 40)
    fake_dialogues = np.repeat(np.arange(10), 10).astype(np.int64)
    
    bank.build(fake_feats, fake_labels, fake_speakers, fake_dialogues)
    
    query = np.random.randn(hidden).astype(np.float32)
    
    feat_g, lab_g, score_g = bank.retrieve_global(query, top_k=3)
    print(f"  global:   feat={feat_g.shape}, labels={lab_g}, scores={score_g}")
    
    feat_s, lab_s, score_s = bank.retrieve_by_speaker(query, 'Monica', top_k=3)
    print(f"  speaker:  feat={feat_s.shape}, labels={lab_s}, scores={score_s}")
    
    feat_d, lab_d, score_d = bank.retrieve_by_dialogue(query, 0, top_k=3)
    print(f"  dialogue: feat={feat_d.shape}, labels={lab_d}, scores={score_d}")
    
    print("\n✓ Emotion-RAG 所有模块验证通过!")
