"""
Phase 2: Seq2Seq + Bahdanau Attention

改动最小化：Encoder 不变，Decoder 每步算 attention + 拼接 context
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'phase2_bahdanau'))
from phase1_1_lstm.model import Encoder
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
        # enc_out: (B, S, H*2), hidden: (num_layers, B, H)

        embedded = self.embed(tgt)  # (B, T, E)
        src_mask = (enc_out.sum(-1) == 0)  # (B, S)

        # Step 1: Get LSTM decoder outputs for all time steps at once
        # First, run LSTM on just the embeddings (no context yet)
        # to get rough decoder states for attention computation
        dec_outputs, (hidden, cell) = self.rnn(
            torch.cat([embedded, torch.zeros(embedded.size(0), embedded.size(1), enc_out.size(-1), device=embedded.device)], dim=-1),
            (hidden, cell)
        )  # (B, T, H)

        # Step 2: Compute attention for ALL time steps in parallel
        context, _ = self.attention(dec_outputs, enc_out, src_mask)  # (B, T, H*2)

        # Step 3: Re-run LSTM with context concatenated (teacher forcing in parallel)
        rnn_input = torch.cat([embedded, context], dim=-1)  # (B, T, E+H*2)
        dec_outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # (B, T, H)

        logits = self.out(dec_outputs)  # (B, T, V)
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
