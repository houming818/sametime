"""
Phase 1: RNN Seq2Seq（无 attention）

Encoder: BiLSTM
Decoder: LSTM
训练: teacher forcing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size)  # 合并双向 → decoder hidden_size

    def forward(self, src, src_len):
        # src: (B, S)   src_len: (B,)
        embedded = self.embed(src)                     # (B, S, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # enc_out: (B, S, H*2)
        # hidden: (2*num_layers, B, H)  — 前向后向各一层
        # cell:   (2*num_layers, B, H)

        # 将双向 hidden 合并为单向 (num_layers, B, H)
        hidden = hidden.view(2, 2, -1, hidden.size(2))  # (2_dir, num_layers, B, H)
        hidden = hidden.sum(dim=0)                       # (num_layers, B, H)
        cell = cell.view(2, 2, -1, cell.size(2)).sum(dim=0)

        return enc_out, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, encoder_states):
        # tgt: (B, T)
        enc_out, (hidden, cell) = encoder_states
        embedded = self.embed(tgt)       # (B, T, E)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # (B, T, H)
        logits = self.out(output)        # (B, T, V)
        return logits, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, hidden = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt, (enc_out, hidden))
        return logits
