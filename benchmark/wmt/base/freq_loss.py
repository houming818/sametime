"""
base/freq_loss.py — Frequency-domain loss via DCT

Idea: Apply DCT to the token probability sequence, penalize
high-frequency noise. This regularizes the model to produce
smoother, more translation-like outputs across the time axis.

HM theory: IFFT of discrete BLEU creates a continuous, differentiable
surface. The frequency components encode multi-scale structure.
"""

import torch
import torch.nn.functional as F
import math


def dct_1d(x, norm='ortho'):
    """
    1D Discrete Cosine Transform (Type II).
    Fully differentiable via torch.matmul.

    x: (B, T)  — any 1D signal per sample
    Returns: (B, T) DCT coefficients
    """
    B, T = x.shape
    device = x.device

    # DCT-II matrix
    k = torch.arange(T, dtype=torch.float32, device=device)
    n = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    D = torch.cos(math.pi / T * (n + 0.5) * k)  # (T, T)

    if norm == 'ortho':
        D[0, :] *= 1.0 / math.sqrt(2)
        D *= math.sqrt(2.0 / T)

    return x @ D.T  # (B, T)


def freq_smoothness_loss(logits, pad_id=0, high_pass_ratio=0.3):
    """
    Frequency-domain smoothness loss.

    Penalizes high-frequency energy in the token probability
    distribution along the time axis. Encourages the model to
    produce smoother transitions between tokens.

    Args:
      logits: (B, T, V)
      pad_id: PAD token id
      high_pass_ratio: fraction of highest frequencies to penalize

    Returns:
      scalar loss
    """
    B, T, V = logits.shape
    device = logits.device

    probs = F.softmax(logits, dim=-1)  # (B, T, V)

    # Get "mass on most-likely token" at each position → (B, T)
    confidence = probs.max(dim=-1).values  # (B, T)

    # DCT along time axis → frequency domain
    freq = dct_1d(confidence)  # (B, T)

    # Penalize high frequencies
    cutoff = max(1, int(T * (1 - high_pass_ratio)))
    high_freq_energy = freq[:, cutoff:].pow(2).mean()

    return high_freq_energy


def freq_bleu_loss(logits, ref_ids, pad_id=0, eos_id=2, ce_weight=0.8, freq_weight=0.1):
    """
    Combined: CE + frequency smoothness regularization.

    The CE handles token accuracy. The frequency regularization
    encourages the model to produce smoother probability distributions,
    which should reduce overfitting (high-frequency jitter).
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ref_ids.reshape(-1),
        ignore_index=pad_id
    )

    freq_loss = freq_smoothness_loss(logits[:, :-1, :], pad_id)

    total = ce_weight * ce + freq_weight * freq_loss
    return total, ce.detach(), freq_loss.detach()
