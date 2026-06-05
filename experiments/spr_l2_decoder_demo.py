import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = 128
V = 16000
td = 5

# 1. 冻结的 L0-L1 渲染管道 (模拟从 Phase 2 加载)
class FrozenL1Encoder(nn.Module):
    def __init__(self, ckpt_path='/mnt/nas/datasets/wmt17/checkpoints/anchor_tree_nce.pt'):
        super().__init__()
        self.L0 = nn.Embedding(V, d)
        self.t_nodes = nn.ModuleList([nn.Embedding(2**i, d) for i in range(td)])
        self.t_merge = nn.Linear(d, d)
        
        # 尝试加载真实权重
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            self.L0.load_state_dict(ckpt['L0'])
            for i in range(td):
                self.t_nodes[i].load_state_dict(ckpt['t_nodes'][i])
            self.t_merge.load_state_dict(ckpt['t_merge'])
            print(f"✅ 成功加载 Phase 2 L1 检查点: {ckpt_path}")
        except Exception as e:
            print(f"⚠️ 无法加载检查点，使用随机初始化: {e}")

        # 核心：L0 和 L1 在 Phase 3 初期被冻结！只训练语序渲染。
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, tok_ids):
        # L0 拓扑底座映射
        t = F.normalize(self.L0(tok_ids), dim=-1)
        
        # L1 Context 极坐标引流场 (这里使用简化的独立树路径模拟)
        w = torch.zeros(*tok_ids.shape, d, device=tok_ids.device)
        for l in range(td):
            nidx = torch.clamp(tok_ids // (V // (2 ** l)), 0, (2 ** l) - 1) if l > 0 else torch.zeros_like(tok_ids)
            w = w + self.t_nodes[l](nidx)
        w = F.normalize(w, dim=-1)
        
        # 复平面渲染 (Complex Multiply)
        tL, tR = t[..., :d//2], t[..., d//2:]
        wL, wR = w[..., :d//2], w[..., d//2:]
        out = self.t_merge(torch.cat([tL*wL - tR*wR, tL*wR + tR*wL], -1))
        
        return out # [Batch, Source_SeqLen, D]

# 2. L2 语序重排与结构词渲染器
class DecoderL2(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, d_model) * 0.05) # 绝对位置编码
        
        # 使用 TransformerDecoder 来实现 Cross-Attention 重排
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=512)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, V)

    def forward(self, tgt_ids, l1_memory):
        # tgt_ids: [Batch, Target_SeqLen]
        # l1_memory: [Batch, Source_SeqLen, D]
        seq_len = tgt_ids.size(1)
        tgt_emb = self.emb(tgt_ids) + self.pos_enc[:, :seq_len, :]
        
        # 典型的自回归 Causal Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_ids.device)
        
        # 解码与重排！(L1 流形在这里被 Cross-Attention 重塑)
        out = self.transformer_decoder(tgt_emb, l1_memory, tgt_mask=tgt_mask)
        
        return self.fc_out(out) # [Batch, Target_SeqLen, Vocab_Size]

if __name__ == "__main__":
    print("=== 初始化 SPR L0-L1-L2 Pipeline ===")
    encoder = FrozenL1Encoder().to(device)
    decoder = DecoderL2(d_model=128, nhead=4, num_layers=2).to(device)

    # 构造假数据 (Batch Size = 2)
    # 源语言 EN
    src_en = torch.randint(3, 1000, (2, 8)).to(device) 
    
    # 目标语言 ZH
    tgt_zh = torch.randint(3, 1000, (2, 12)).to(device)
    tgt_input = tgt_zh[:, :-1]
    tgt_label = tgt_zh[:, 1:]

    print(f"源语言 (EN) 形状: {src_en.shape}")
    print(f"目标输入 (ZH) 形状: {tgt_input.shape}")

    # 前向传播 (Forward)
    print("\n--- 渲染流开始 ---")
    l1_features = encoder(src_en)
    print(f"L1 渲染流形特征: {l1_features.shape}  <-- 这里包含了纯净且消歧后的词义，但依然是英文语序")

    logits = decoder(tgt_input, l1_features)
    print(f"L2 生成概率分布: {logits.shape}  <-- 这里已经过重排，输出了中文句法空间的 Token 分布")

    # 损失计算
    loss = F.cross_entropy(logits.reshape(-1, V), tgt_label.reshape(-1))
    print(f"\n当前未训练交叉熵损失: {loss.item():.4f}")

    # 反向传播测试 (Backward)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    
    # 确保 L1 梯度为 0 (被冻结)
    l1_grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
    print(f"L1 编码器梯度范数: {l1_grad_norm} (预期为 0，因为被冻结)")
    
    optimizer.step()
    print("✅ 反向传播通过，L2 Decoder 重排训练循环连通！")
