import torch
import torch.nn as nn
import torch.nn.functional as F

class HeapTreeLayer(nn.Module):
    """
    语义前缀路由的核心层 (Heap Tree Layer)。
    包含底层的 Flat Embedding (L0) 以及一棵 Heap Tree。
    Tree 提取共享路径的旋转/放缩因子，然后通过“单层复数乘法”与 L0 结合。
    """
    def __init__(self, vocab_size, embed_dim, depth=5, agg_method='complex_mul'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.agg_method = agg_method
        
        # 基础 Token 嵌入 (平坦世界，保留独立性)
        self.L0 = nn.Embedding(vocab_size, embed_dim)
        
        # 树的各层节点 (每层 2^i 个节点)
        self.t_nodes = nn.ModuleList([
            nn.Embedding(2 ** i, embed_dim) for i in range(depth)
        ])
        
        # 为了稳定训练，初始化 L0 和树节点
        nn.init.normal_(self.L0.weight, 0, 0.02)
        for tn in self.t_nodes:
            nn.init.normal_(tn.weight, 0, 0.02)
            
        if self.agg_method == 'mlp_add':
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )

    def get_tree_vec(self, token_ids):
        """
        计算 Token 在树上路径的累加向量 (Tree Vector)，支持任意形状输入
        """
        input_shape = token_ids.shape
        device = token_ids.device
        w = torch.zeros(*input_shape, self.embed_dim, device=device)
        
        for level in range(self.depth):
            num_branches = 2 ** level
            stride = max(1, self.vocab_size // num_branches)
            # 计算在该层的节点 index
            nidx = torch.clamp(token_ids // stride, 0, num_branches - 1)
            w = w + self.t_nodes[level](nidx)
            
        return w

    def forward(self, token_ids):
        # 1. 获取平坦世界的独立特征 t
        t = self.L0(token_ids)
        
        # 2. 获取树上的共享特征 w (路径上的所有节点相加)
        w = self.get_tree_vec(token_ids)
        
        # 为了保证稳定，按论文惯例我们做一下 normalize 也可以，但 Echo 阶段先做基础聚合
        # 3. 将 t 和 w 聚合
        if self.agg_method == 'complex_mul':
            # 将 t 和 w 分为实部和虚部 (假设 embed_dim 是偶数)
            d2 = self.embed_dim // 2
            tL, tR = t[..., :d2], t[..., d2:]
            wL, wR = w[..., :d2], w[..., d2:]
            
            # 复数乘法: (tL + i tR) * (wL + i wR)
            # 实部: tL*wL - tR*wR
            # 虚部: tL*wR + tR*wL
            out_real = tL * wL - tR * wR
            out_imag = tL * wR + tR * wL
            return torch.cat([out_real, out_imag], dim=-1)
            
        elif self.agg_method == 'simple_add':
            # 基础对照组：直接相加
            return t + w
            
        elif self.agg_method == 'mlp_add':
            # 深度/非线性对照组：拼接后过 MLP
            combined = torch.cat([t, w], dim=-1)
            return self.mlp(combined)
            
        else:
            raise ValueError(f"不支持的聚合方法: {self.agg_method}")
