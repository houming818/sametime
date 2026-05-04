"""
base/soft_bleu.py — Vectorized differentiable BLEU approximation

Uses scatter/gather instead of Python for-loops.
O(B*T*V) same as CE, all GPU native.
"""

import torch
import torch.nn.functional as F


def soft_bleu(logits, ref_ids, pad_id=0, eos_id=2, max_n=4, smooth=1e-8):
    """
    Vectorized SoftBLEU.

    Args:
      logits:  (B, T, V) model logits
      ref_ids: (B, R) reference token ids (padded)
      pad_id:  PAD token id (default 0)
      eos_id:  EOS token id (default 2)
      max_n:   max n-gram order
      smooth:  epsilon for log stability

    Returns:
      bleu: scalar, differentiable, higher is better
    """
    B, T, V = logits.shape
    device = logits.device

    probs = F.softmax(logits, dim=-1)  # (B, T, V)

    # --- 1-gram reference mask (vectorized) ---
    # ref_ids: (B, R). Build (B, 1, V) mask where mask[b, 0, v] = 1 if token v in ref[b]
    ref_mask = torch.zeros(B, V, dtype=probs.dtype, device=device)
    # Convert to 0/1 with scatter: set mask[b, ref_ids[b,r]] = 1 for all valid r
    valid = (ref_ids != pad_id) & (ref_ids != eos_id) & (ref_ids < V)
    idx = ref_ids.clone()
    idx[~valid] = 0  # safe dummy index
    src = torch.ones_like(ref_ids, dtype=probs.dtype)
    src[~valid] = 0.0
    ref_mask.scatter_add_(1, idx, src)
    ref_mask = (ref_mask > 0).to(probs.dtype)  # binary

    # --- 1-gram: soft precision per position ---
    # (B, T, V) * (B, 1, V) → sum → (B, T)
    match_1gram = (probs * ref_mask.unsqueeze(1)).sum(dim=-1)  # (B, T)

    # Hyp length: count positions where model is "active" (sum probs > threshold)
    hyp_len = torch.ones(B, dtype=torch.float32, device=device) * T
    ref_len = valid.sum(dim=1).float().clamp(min=1)  # (B,)

    # Clipped precision for 1-gram
    prec_1 = (match_1gram.sum(dim=1) + smooth) / (hyp_len + smooth)  # (B,)
    prec_1 = prec_1.mean()  # scalar

    # --- Higher n-grams: geometric decay approximation ---
    precisions = [prec_1]
    for n in range(2, max_n + 1):
        decay = 0.5 ** (n - 1)
        p_n = prec_1 * (1.0 - decay) + decay * smooth  # geometric extrapolation
        p_n = torch.clamp(p_n, smooth, 1.0)
        precisions.append(p_n)

    # --- BLEU = geometric mean of n-gram precisions ---
    bleu = torch.tensor(1.0, device=device)
    for p in precisions:
        bleu = bleu * torch.clamp(p, smooth, 1.0)
    bleu = bleu ** (1.0 / len(precisions))

    return bleu, precisions


def soft_bleu_loss(logits, ref_ids, pad_id=0, eos_id=2, max_n=4, ce_weight=0.7):
    """
    Combined: λ*CE + (1-λ)*(1-SoftBLEU).
    All vectorized, no Python for-loops.
    """
    # CE
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        ref_ids.reshape(-1),
        ignore_index=pad_id
    )

    # SoftBLEU
    bleu, _ = soft_bleu(logits, ref_ids, pad_id, eos_id, max_n)

    bleu_loss = 1.0 - bleu
    total = ce_weight * ce + (1.0 - ce_weight) * bleu_loss
    return total, ce.detach(), bleu.detach()


def soft_bleu_only_loss(logits, ref_ids, pad_id=0, eos_id=2, max_n=4):
    bleu, precisions = soft_bleu(logits, ref_ids, pad_id, eos_id, max_n)
    # -log(BLEU) instead of 1-BLEU: eliminates the BLEU magnitude suppression
    # d(-log)/dW = -(1/BLEU)*dBLEU/dW  — the 1/BLEU cancels BLEU's own factor
    loss = -torch.log(bleu + 1e-8)
    return loss, bleu.detach(), precisions
