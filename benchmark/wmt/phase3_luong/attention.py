"""
Phase 3: Luong Attention（dot / general / concat）

与 Bahdanau 的区别：
  - 在 decoder output 上计算（而非 hidden state）
  - 更简单的 score 函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super().__init__()
        self.method = method
        if method == "general":
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        elif method == "concat":
            self.W = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def score(self, decoder_output, encoder_outputs):
        # decoder_output: (B, H)
        # encoder_outputs: (B, S, H*2)
        if self.method == "dot":
            # 需要将 decoder H 投影到 H*2 或只取前 H 维
            return torch.bmm(encoder_outputs, decoder_output.unsqueeze(-1)).squeeze(-1)
        elif self.method == "general":
            return torch.bmm(self.W(encoder_outputs), decoder_output.unsqueeze(-1)).squeeze(-1)
        elif self.method == "concat":
            dec = decoder_output.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
            return self.v(torch.tanh(self.W(torch.cat([encoder_outputs, dec], dim=-1)))).squeeze(-1)

    def forward(self, decoder_output, encoder_outputs, src_mask=None):
        energy = self.score(decoder_output, encoder_outputs)  # (B, S)
        if src_mask is not None:
            energy = energy.masked_fill(src_mask, -1e9)
        attn = F.softmax(energy, dim=-1)                     # (B, S)
        context = (attn.unsqueeze(-1) * encoder_outputs).sum(dim=1)  # (B, H*2)
        return context, attn
