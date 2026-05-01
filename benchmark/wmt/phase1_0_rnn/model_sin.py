"""
Phase 1.0-sin: RNN with sin() activation
========

Replaces tanh with sin in hidden state computation:

  h_t = sin(W_hh·h_{t-1} + W_ih·x_t + b)

Key difference from tanh:
  - sin is periodic: same hidden value can arise from infinite different inputs
  - Each monotonic interval acts as a separate "hash bucket"
  - Gradient = cos(pre_activation) → oscillates, can resurrect at later steps
  - Output range [-1, 1] (same as tanh), but mapping is many-to-many

Hypothesis:
  tanh:   single hash bucket → all inputs collide to (-1, +1) → BLEU ceiling ~3.0
  sin:    infinite hash buckets → period selection adds one more degree of freedom
"""

import torch
import torch.nn as nn


class SinCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        # h_t = sin(W_hh·h_{t-1} + W_ih·x_t + b)
        return torch.sin(self.W_hh(h) + self.W_ih(x))


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Bidirectional: each direction gets its own set of cells
        self.cell_fw = nn.ModuleList([
            SinCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.cell_bw = nn.ModuleList([
            SinCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, src, src_len):
        # src: (B, S)
        B, S = src.shape
        embedded = self.dropout(self.embed(src))  # (B, S, E)

        h_fw = [torch.zeros(B, self.hidden_size, device=src.device) for _ in range(self.num_layers)]
        h_bw = [torch.zeros(B, self.hidden_size, device=src.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(S):
            # forward direction
            x_t = embedded[:, t, :]
            for l in range(self.num_layers):
                inp = x_t if l == 0 else self.dropout(h_fw[l-1])
                h_fw[l] = self.cell_fw[l](inp, h_fw[l])

            # backward direction
            tb = S - 1 - t
            x_b = embedded[:, tb, :]
            for l in range(self.num_layers):
                inp = x_b if l == 0 else self.dropout(h_bw[l-1])
                h_bw[l] = self.cell_bw[l](inp, h_bw[l])

            # concat forward + backward top layer
            output_t = torch.cat([h_fw[-1], h_bw[-1]], dim=1)  # (B, 2H)
            outputs.append(output_t)

        enc_out = torch.stack(outputs, dim=1)  # (B, S, 2H)

        # Final hidden for decoder: concat last step fw+bw → sum to single direction
        hidden = (h_fw[-1] + h_bw[-1]) / 2  # (B, H)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (N, B, H)

        return enc_out, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.cells = nn.ModuleList([
            SinCell(embed_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden):
        # tgt: (B, T), hidden: (N, B, H)
        B, T = tgt.shape
        embedded = self.dropout(self.embed(tgt))  # (B, T, E)

        h = [hidden[i] for i in range(self.num_layers)]  # list of (B, H)
        logits_list = []

        for t in range(T):
            x_t = embedded[:, t, :]
            for l in range(self.num_layers):
                inp = x_t if l == 0 else self.dropout(h[l-1])
                h[l] = self.cells[l](inp, h[l])
            logits_list.append(self.out(h[-1]))

        logits = torch.stack(logits_list, dim=1)  # (B, T, V)
        return logits, torch.stack(h, dim=0)  # (N, B, H)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out, hidden = self.encoder(src, src_len)
        logits, _ = self.decoder(tgt, hidden)
        return logits
