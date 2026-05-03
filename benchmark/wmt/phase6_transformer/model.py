"""
Phase 6: Transformer（Vaswani 2017）

最小可运行实现，不依赖 fairseq/transformers。

关键 hyperparams（论文 base config 缩放到 IWSLT14）：
- d_model=256, num_heads=4, d_ff=1024, num_layers=3, dropout=0.1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — 不需要学习"""
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ═══════════════════════════════════════════
# Multi-Head Attention
# ═══════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # Q/K/V: (B, T, d_model)
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.W_o.out_features)
        return self.W_o(out)


# ═══════════════════════════════════════════
# Feed-Forward Network
# ═══════════════════════════════════════════

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ═══════════════════════════════════════════
# Encoder Layer + Encoder
# ═══════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_heads=4,
                 d_ff=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_len):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # (B, S, d_model)


# ═══════════════════════════════════════════
# Decoder Layer + Decoder
# ═══════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # self-attention (masked)
        x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        # cross-attention (decoder → encoder)
        x = x + self.dropout2(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        # FFN
        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_heads=4,
                 d_ff=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, src_len):
        # causal mask（确保 decoder 不能看到未来 token）
        T = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(1, 1, T, T, device=tgt.device))  # (1, 1, T, T)

        src_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.out(self.norm(x))  # (B, T, V)


# ═══════════════════════════════════════════
# Transformer (Encoder-Decoder)
# ═══════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_len):
        enc_out = self.encoder(src, src_len)
        logits = self.decoder(tgt, enc_out, src_len)
        return logits
