"""
Phase 3: Seq2Seq + Luong Attention + Beam Search
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from phase1_rnn.model import Encoder
from attention import LuongAttention


class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2,
                 dropout=0.3, attn_method="general"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.attention = LuongAttention(hidden_size, method=attn_method)
        # 拼接 context 后投影回 hidden_size
        self.W_c = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, encoder_states):
        enc_out, (hidden, cell) = encoder_states
        embedded = self.embed(tgt)                          # (B, T, E)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # (B, T, H)

        # per-step attention (简化：用整个序列的 output)
        # 实际 Luong 论文中每个 step 独立计算；这里为简洁，取最后一步
        dec_out = output[:, -1, :]                          # (B, H)
        src_mask = (enc_out.sum(-1) == 0)                   # (B, S)
        context, _ = self.attention(dec_out, enc_out, src_mask)  # (B, H*2)

        # 拼接 + 投影
        concat = torch.cat([dec_out, context], dim=-1)       # (B, H + H*2)
        hidden_state = torch.tanh(self.W_c(concat))          # (B, H)
        logits = self.out(hidden_state).unsqueeze(1)         # (B, 1, V)
        return logits, (enc_out, (hidden, cell))


class LuongSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, hidden = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt, (enc_out, hidden))
        return logits
