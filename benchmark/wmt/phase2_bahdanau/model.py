"""
Phase 2: Seq2Seq + Bahdanau Attention

改动最小化：Encoder 不变，Decoder 每步算 attention + 拼接 context
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from phase1_rnn.model import Encoder
from attention import BahdanauAttention


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = BahdanauAttention(hidden_size)
        # 输入 = [embed; context] → embed_size + hidden_size*2
        self.rnn = nn.LSTM(embed_size + hidden_size * 2, hidden_size,
                           num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, encoder_states):
        enc_out, (hidden, cell) = encoder_states
        # enc_out: (B, S, H*2)
        # hidden:  (num_layers, B, H)

        embedded = self.embed(tgt)            # (B, T, E)
        outputs = []
        for t in range(embedded.size(1)):
            # 上一步 hidden[-1] 作为 decoder state
            dec_h = hidden[-1]                # (B, H)
            src_mask = (enc_out.sum(-1) == 0)  # (B, S) — padding mask
            context, _ = self.attention(dec_h, enc_out, src_mask)   # (B, H*2)

            rnn_input = torch.cat([embedded[:, t, :], context], dim=-1).unsqueeze(1)  # (B, 1, E+H*2)
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            outputs.append(output)

        logits = self.out(torch.cat(outputs, dim=1))  # (B, T, V)
        return logits, (hidden, cell)


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, hidden = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt, (enc_out, hidden))
        return logits
