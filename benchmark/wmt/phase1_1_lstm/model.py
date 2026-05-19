"""
Phase 1.1: LSTM Seq2Seq（三扇门 + 细胞状态）
========

Encoder: BiLSTM
Decoder: LSTM
训练: teacher forcing

LSTM 用三个门（遗忘门 f_t、输入门 i_t、输出门 o_t）加上细胞状态 c_t，
解决了 vanilla RNN 的梯度消失问题：

  f_t = σ(W_f·[h_{t-1}, x_t])      # 遗忘门：丢掉多少旧记忆
  i_t = σ(W_i·[h_{t-1}, x_t])      # 输入门：写入多少新信息
  o_t = σ(W_o·[h_{t-1}, x_t])      # 输出门：暴露多少给外面
  c̃_t = tanh(W_c·[h_{t-1}, x_t])   # 候选记忆

  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t   # 细胞状态：加性更新 → 梯度直通
  h_t = o_t ⊙ tanh(c_t)             # 隐藏输出

关键：c_t 的更新是 f_t*c_{t-1} + i_t*c̃_t，没有连乘 tanh，
梯度可以通过 c_{t-1}→c_t 的"高速公路"直达早期时间步。
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           bidirectional=True, dropout=dropout, batch_first=True)

    def forward(self, src, src_len):
        embedded = self.embed(src)
        src_len_sorted, sort_idx = src_len.sort(0, descending=True)
        _, unsort_idx = sort_idx.sort(0)
        embedded_sorted = embedded[sort_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded_sorted, src_len_sorted.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (hidden, cell) = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        enc_out = enc_out[unsort_idx]
        unsort_dev = unsort_idx.to(hidden.device)
        hidden = hidden.index_select(1, unsort_dev)
        cell = cell.index_select(1, unsort_dev)

        hidden = hidden.view(2, self.rnn.num_layers, -1, hidden.size(2)).sum(dim=0)
        cell = cell.view(2, self.rnn.num_layers, -1, cell.size(2)).sum(dim=0)

        return enc_out, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, encoder_states):
        enc_out, (hidden, cell) = encoder_states
        embedded = self.embed(tgt)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.out(output)
        return logits, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, states = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt, states)
        return logits
