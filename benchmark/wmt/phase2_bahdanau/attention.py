"""
Phase 2: Bahdanau Additive Attention

公式: e_ij = v^T tanh(W s_i + U h_j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)   # U_a: project encoder outputs
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)   # W_a: project decoder states
        self.v = nn.Linear(hidden_size, 1, bias=False)             # v_a: score projection

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        """
        decoder_hidden:  (B, H)           — 当前 decoder hidden state
        encoder_outputs: (B, S, H*2)      — 双向 encoder 各时刻输出
        src_mask:        (B, S)           — padding 位置为 True

        return:
          context: (B, H*2) — 加权求和后的上下文向量
          attn:    (B, S)   — 注意力权重（可用于可视化）
        """
        # encoder_outputs 是双向的 (H*2)，需要投影到 H
        # 先线性变换 encoder_outputs
        enc_proj = self.W(encoder_outputs)          # (B, S, H)

        # decoder hidden 扩展并与 enc_proj 相加
        dec_proj = self.U(decoder_hidden).unsqueeze(1)  # (B, 1, H)
        score = torch.tanh(enc_proj + dec_proj)         # (B, S, H)
        energy = self.v(score).squeeze(-1)              # (B, S)

        # mask padding 位置
        if src_mask is not None:
            energy = energy.masked_fill(src_mask, -1e9)
        attn = F.softmax(energy, dim=-1)               # (B, S)

        # 上下文向量 = 加权求和
        context = (attn.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # (B, H*2)
        return context, attn
