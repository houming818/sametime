"""
base/bleu_function.py — BLEU as torch.autograd.Function

Forward: 真实 BLEU (硬 argmax + n-gram 计数)
Backward: ReLU 风格 0/1 信号——匹配得 1, 不匹配得 0
          通过 softmax 概率分布回传到 logits。
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function


def _ngram_counts(tokens, n):
    """Count n-grams in token sequence (list or tensor)."""
    counts = {}
    tok_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    for i in range(len(tok_list) - n + 1):
        gram = tuple(tok_list[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _compute_bleu(hyp_tokens, ref_tokens, max_n=4):
    """Compute true BLEU-4 for one sentence pair. Returns float."""
    brevity = min(1.0, len(hyp_tokens) / max(1, len(ref_tokens)))

    log_bleu = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _ngram_counts(hyp_tokens, n)
        ref_ng = _ngram_counts(ref_tokens, n)
        clipped = sum(min(hyp_ng.get(g, 0), ref_ng.get(g, 0)) for g in hyp_ng)
        total = max(1, len(hyp_tokens) - n + 1)
        prec = clipped / total if total > 0 else 0.0
        if prec > 0:
            log_bleu += torch.log(torch.tensor(prec))
        else:
            return 0.0

    bleu = brevity * torch.exp(log_bleu / max_n)
    return float(bleu)


class BLEUFunction(Function):
    """
    BLEU as differentiable function.

    Forward: compute true BLEU (for logging/reporting).
    Backward: 0/1 signal — for each position, gradient=1 if the argmax
              token appears in the reference, else 0. Gradient is
              distributed through softmax to logits.
    """

    @staticmethod
    def forward(ctx, logits, ref_ids, pad_id=0, eos_id=2, max_n=4):
        """
        logits: (B, T, V)  model logits
        ref_ids: (B, R)   reference token ids (padded)
        Returns: (B,) BLEU per sample (detached)
        """
        B, T, V = logits.shape
        device = logits.device

        # Softmax for backward use
        probs = F.softmax(logits, dim=-1)
        ctx.save_for_backward(probs, ref_ids)
        ctx.pad_id = pad_id
        ctx.eos_id = eos_id

        # Hard argmax for forward BLEU
        pred_tokens = logits.argmax(dim=-1)  # (B, T)

        bleus = []
        ctx.pred_tokens = []
        ctx.ref_lists = []

        for b in range(B):
            # Build reference list (exclude PAD/EOS)
            ref_list = []
            for r in ref_ids[b]:
                r = r.item()
                if r in (pad_id, eos_id):
                    break
                ref_list.append(r)

            # Build hypothesis list (stop at first EOS, exclude PAD)
            hyp_list = []
            for t in range(T):
                tok = pred_tokens[b, t].item()
                if tok == eos_id:
                    break
                if tok != pad_id:
                    hyp_list.append(tok)

            ctx.pred_tokens.append(pred_tokens[b].detach().cpu())
            ctx.ref_lists.append(ref_list)
            bleu = _compute_bleu(hyp_list, ref_list, max_n)
            bleus.append(bleu)

        return torch.tensor(bleus, device=device)

    @staticmethod
    def backward(ctx, grad_output):
        probs, ref_ids = ctx.saved_tensors
        B, T, V = probs.shape
        pad_id = ctx.pad_id
        eos_id = ctx.eos_id
        device = probs.device

        grad_logits = torch.zeros_like(probs)

        for b in range(B):
            ref_set = set()
            for r in ref_ids[b]:
                r = r.item()
                if r in (pad_id, eos_id):
                    break
                ref_set.add(r)
            if not ref_set:
                continue

            # 0/1 signal per position: is the max-prob token in ref?
            pred = probs[b].argmax(dim=-1)  # (T,)
            signal = torch.zeros(T, device=device)
            for t in range(T):
                if pred[t].item() in ref_set:
                    signal[t] = 1.0

            # Distribute signal through softmax to logits
            grad_token = signal.unsqueeze(-1) * probs[b]  # (T, V)
            grad_token = grad_token - grad_token.mean(dim=-1, keepdim=True)
            grad_logits[b] = grad_token * grad_output[b]

        return grad_logits, None, None, None, None


def logits_argmax(logits):
    """Helper for argmax on a single sample."""
    return logits.argmax(dim=-1)


def bleu_loss(logits, ref_ids, pad_id=0, eos_id=2, max_n=4):
    """
    BLEU Function loss: L = 1 - BLEU.
    Fully differentiable via custom backward.
    """
    bleu = BLEUFunction.apply(logits, ref_ids, pad_id, eos_id, max_n)
    return (1.0 - bleu).mean(), bleu.mean().detach()
