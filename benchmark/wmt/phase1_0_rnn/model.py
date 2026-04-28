"""
Phase 1.0: Vanilla RNN Seq2Seq（无门控）
========

Encoder: 双向 RNN (tanh)
Decoder: 单向 RNN (tanh)
训练: teacher forcing

RNN 是 LSTM 的前身。没有遗忘门、输入门、输出门——每个时间步的隐状态
完全由上一隐状态 + 当前输入决定，没有"细胞状态"提供梯度高速通道。

公式:
  h_t = tanh(W_hh · h_{t-1} + W_ih · x_t + b)

致命弱点: 长句中梯度通过 tanh(W) 连乘 → 指数衰减 → 梯度消失。
这也是 Phase 1.1 引入 LSTM 的直接原因。
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 用 nn.RNN 代替 LSTM
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)

    def forward(self, src, src_len):
        # src: (B, S)
        embedded = self.embed(src)                      # (B, S, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        # RNN 只返回 (output, hidden) —— 没有 cell state
        packed_out, hidden = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # enc_out: (B, S, H*2)
        # hidden: (D*num_layers, B, H)  where D=2 (bidirectional)

        # 将双向 hidden 合并为单向 (num_layers, B, H)
        hidden = hidden.view(2, 2, -1, hidden.size(2))  # (2_dir, num_layers, B, H)
        hidden = hidden.sum(dim=0)                       # (num_layers, B, H)

        return enc_out, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers,
                          dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden, enc_out=None):
        # tgt: (B, T)
        # hidden: (num_layers, B, H)  — 只有 hidden，无 cell
        embedded = self.embed(tgt)                       # (B, T, E)
        output, hidden = self.rnn(embedded, hidden)      # (B, T, H), (num_layers, B, H)
        logits = self.out(output)                        # (B, T, V)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, hidden = self.encoder(src, src_len)
        # RNN decoder 只传 hidden，不传 enc_out（无 attention）
        logits, _ = self.decoder(tgt, hidden)
        return logits
