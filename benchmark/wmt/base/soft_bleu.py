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
    valid = (ref_ids != pad_id) & (ref_ids != eos_id) & (ref_ids < V)
    idx = ref_ids.clone()
    idx[~valid] = 0
    src = torch.ones_like(ref_ids, dtype=probs.dtype)
    src[~valid] = 0.0
    ref_mask = torch.zeros(B, V, dtype=probs.dtype, device=device)
    ref_mask.scatter_add_(1, idx, src)
    ref_mask = (ref_mask > 0).to(probs.dtype)

    # --- precisions for all n-gram orders ---
    precisions = _soft_precisions(probs, ref_ids, valid, max_n, pad_id, eos_id, smooth)

    bleu = torch.tensor(1.0, device=device)
    for p in precisions:
        bleu = bleu * torch.clamp(p, smooth, 1.0)
    bleu = bleu ** (1.0 / len(precisions))
    return bleu, precisions


def _soft_precisions(probs, ref_ids, valid, max_n, pad_id, eos_id, smooth):
    """Compute 1-gram through max_n-gram soft precision via scatter_add."""
    B, T, V = probs.shape
    R = ref_ids.size(1)
    device = probs.device

    # 1-gram: bag-of-words match
    ref_mask = torch.zeros(B, V, dtype=probs.dtype, device=device)
    idx = ref_ids.clone(); idx[~valid] = 0
    src = torch.ones_like(ref_ids, dtype=probs.dtype); src[~valid] = 0.0
    ref_mask.scatter_add_(1, idx, src)
    ref_mask = (ref_mask > 0).to(probs.dtype)
    match_1gram = (probs * ref_mask.unsqueeze(1)).sum(dim=-1)  # (B, T)
    prec_1 = (match_1gram.sum(dim=1) + smooth) / (T + smooth)
    prec_1 = prec_1.mean()

    precisions = [prec_1]

    # Higher n-grams: scatter_add over sliding windows
    ref_len = valid.sum(dim=1)  # (B,)
    for n in range(2, max_n + 1):
        total_match = torch.tensor(0.0, device=device)
        total_windows = torch.tensor(0, device=device)
        for b in range(B):
            L = int(ref_len[b].item())
            if L < n or T < n:
                continue
            ref_b = ref_ids[b, :L]  # (L,)
            ngram_idx = ref_b.unfold(0, n, 1)  # (L-n+1, n)
            valid_ng = ((ngram_idx != pad_id) & (ngram_idx != eos_id)).all(dim=1)
            ngram_idx = ngram_idx[valid_ng]  # (N, n)
            if ngram_idx.size(0) == 0:
                continue

            # Joint probability: prod over n consecutive model positions
            joint = torch.ones(T - n + 1, ngram_idx.size(0), device=device)
            for d in range(n):
                gathered = probs[b, d:T - n + 1 + d].gather(
                    1, ngram_idx[:, d].unsqueeze(0).expand(T - n + 1, -1))
                joint *= gathered
            total_match += joint.sum()
            total_windows += (T - n + 1)

        if total_windows == 0:
            p_n = prec_1 * (0.5 ** (n - 1))  # fallback geometric
        else:
            p_n = (total_match + smooth) / (total_windows * B + smooth)
        precisions.append(p_n.clamp(smooth, 1.0))

    return precisions


def soft_bleu_restricted(logits, ref_ids, pad_id=0, eos_id=2, max_n=4, k=170, smooth=1e-8):
    """
    Restricted-softmax SoftBLEU.
    
    Instead of softmax over full vocab (56K), normalize over top-k + ref tokens
    (~190 tokens). Gradient amplified by ≈300x — matches CE gradient magnitude.
    
    k defaults to 170 ≈ K_lang * |V| = 0.003 * 56652.
    """
    B, T, V = logits.shape
    device = logits.device

    # 1. Top-k per position
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)

    # 2. Initialize restricted logits with -inf (zero probability for excluded tokens)
    restricted = torch.full_like(logits, float('-inf'))
    restricted.scatter_(-1, topk_idx, topk_vals)

    # 3. Reference token mask — union with topk
    valid = (ref_ids != pad_id) & (ref_ids != eos_id) & (ref_ids < V)
    ref_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    idx_safe = ref_ids.clone()
    idx_safe[~valid] = 0
    ref_mask.scatter_(1, idx_safe, valid)

    # Include reference tokens: their own logits replace -inf if not already in topk
    restricted = torch.where(ref_mask.unsqueeze(1), logits, restricted)

    # 4. Restricted softmax (denominator over ~190 tokens, not 56K)
    probs = F.softmax(restricted, dim=-1)

    # 5. Real n-gram precision (same scatter_add as full version)
    precisions = _soft_precisions(probs, ref_ids, valid, max_n, pad_id, eos_id, smooth)

    bleu = torch.tensor(1.0, device=device)
    for p in precisions:
        bleu = bleu * torch.clamp(p, smooth, 1.0)
    bleu = bleu ** (1.0 / len(precisions))
    return bleu, precisions


def soft_bleu_loss(logits, ref_ids, pad_id=0, eos_id=2, max_n=4, ce_weight=0.7):
    """
    Combined: λ*CE + (1-λ)*(1-SoftBLEU).
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
