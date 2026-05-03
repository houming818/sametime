"""
Phase 2: Bahdanau Additive Attention

公式: e_ij = v^T tanh(W s_i + U h_j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, enc_hidden_size=None):
        super().__init__()
        if enc_hidden_size is None:
            enc_hidden_size = hidden_size * 2  # bidirectional encoder output
        self.W = nn.Linear(enc_hidden_size, hidden_size, bias=False)   # U_a: project encoder outputs
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)   # W_a: project decoder states
        self.v = nn.Linear(hidden_size, 1, bias=False)             # v_a: score projection

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        """
        decoder_hidden:  (B, H)  or  (B, T, H)   — 单步或全部时间步
        encoder_outputs: (B, S, H*2)              — 双向 encoder 各时刻输出
        src_mask:        (B, S)                   — padding 位置为 True

        return:
          context: (B, H*2) or (B, T, H*2)
          attn:    (B, S)  or (B, T, S)
        """
        enc_proj = self.W(encoder_outputs)  # (B, S, H)

        if decoder_hidden.dim() == 3:
            # Batch mode: (B, T, H) — compute attention for all time steps in parallel
            B, T, H = decoder_hidden.shape
            dec_proj = self.U(decoder_hidden)  # (B, T, H)
            # enc_proj: (B, S, H) → (B, 1, S, H), dec_proj: (B, T, 1, H)
            score = torch.tanh(enc_proj.unsqueeze(1) + dec_proj.unsqueeze(2))  # (B, T, S, H)
            energy = self.v(score).squeeze(-1)  # (B, T, S)

            if src_mask is not None:
                energy = energy.masked_fill(src_mask.unsqueeze(1), -1e9)
            attn = F.softmax(energy, dim=-1)  # (B, T, S)

            # context: (B, T, H*2)
            context = torch.bmm(attn, encoder_outputs)  # (B, T, H*2)
            return context, attn
        else:
            # Single-step mode: (B, H)
            dec_proj = self.U(decoder_hidden).unsqueeze(1)  # (B, 1, H)
            score = torch.tanh(enc_proj + dec_proj)  # (B, S, H)
            energy = self.v(score).squeeze(-1)  # (B, S)

            if src_mask is not None:
                energy = energy.masked_fill(src_mask, -1e9)
            attn = F.softmax(energy, dim=-1)  # (B, S)

            context = (attn.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # (B, H*2)
            return context, attn
