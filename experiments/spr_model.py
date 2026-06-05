import torch
import torch.nn as nn
import torch.nn.functional as F
from spr_tree_layer import HeapTreeLayer

class SemanticPrefixRoutingModel(nn.Module):
    """
    双语语义前缀路由模型 (Semantic Prefix Routing Model)。
    包装了 HeapTreeLayer，并实现多 Token BPE 平均表示计算及 InfoNCE 对齐。
    """
    def __init__(self, vocab_size, embed_dim, depth=5, agg_method='complex_mul'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.agg_method = agg_method
        
        # 核心 Heap Tree 路由层
        self.tree_layer = HeapTreeLayer(vocab_size, embed_dim, depth, agg_method)
        
        # 融合后的投影层，采用恒等矩阵初始化
        self.t_merge = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.t_merge.weight)
        nn.init.zeros_(self.t_merge.bias)
        
    def get_word_representation(self, padded, mask):
        """
        根据 padded token ID 和对应的归一化 mask，计算其单词级别的流形特征。
        """
        # 1. 获取平坦世界的特征并求平均
        t_tokens = self.tree_layer.L0(padded)  # [N, SeqLen, D]
        t_mean = (t_tokens * mask.unsqueeze(-1)).sum(dim=1)  # [N, D]
        t = F.normalize(t_mean, dim=-1)
        
        # 2. 获取树上的路由特征并求平均
        w_tokens = self.tree_layer.get_tree_vec(padded)  # [N, SeqLen, D]
        w_mean = (w_tokens * mask.unsqueeze(-1)).sum(dim=1)  # [N, D]
        w = F.normalize(w_mean, dim=-1)
        
        # 3. 聚合
        if self.agg_method == 'complex_mul':
            d2 = self.embed_dim // 2
            tL, tR = t[..., :d2], t[..., d2:]
            wL, wR = w[..., :d2], w[..., d2:]
            
            out_real = tL * wL - tR * wR
            out_imag = tL * wR + tR * wL
            
            combined = torch.cat([out_real, out_imag], dim=-1)
            return self.t_merge(combined)
            
        elif self.agg_method == 'simple_add':
            return t + w
            
        elif self.agg_method == 'mlp_add':
            combined = torch.cat([t, w], dim=-1)
            return self.tree_layer.mlp(combined)
            
        else:
            raise ValueError(f"不支持的聚合方法: {self.agg_method}")
            
    def compute_infonce_loss(self, en_padded, en_mask, zh_padded, zh_mask, temp=0.07):
        """
        计算英-中双语锚点对的 InfoNCE 损失。
        """
        # 1. 获取英中单词级特征
        en_vecs = self.get_word_representation(en_padded, en_mask)  # [N, D]
        zh_vecs = self.get_word_representation(zh_padded, zh_mask)  # [N, D]
        
        # 2. L2 归一化后计算余弦相似度矩阵
        en_norm = F.normalize(en_vecs, dim=-1)
        zh_norm = F.normalize(zh_vecs, dim=-1)
        
        # 3. 计算 logits
        logits = (en_norm @ zh_norm.T) / temp  # [N, N]
        
        # 4. 对称 InfoNCE 损失 (双向最大化对齐)
        N = en_padded.size(0)
        labels = torch.arange(N, device=en_padded.device)
        
        loss_en_zh = F.cross_entropy(logits, labels)
        loss_zh_en = F.cross_entropy(logits.T, labels)
        
        return (loss_en_zh + loss_zh_en) / 2.0
