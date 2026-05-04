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
    def forward(ctx, logits, ref_ids, pad_id=0, eos_id=2, max_n=4, invert=False):
        """
        logits: (B, T, V)  model logits
        ref_ids: (B, R)   reference token ids (padded)
        invert: if True, signal=1 for wrong tokens (inverted hash hypothesis)
        Returns: (B,) BLEU per sample (detached)
        """
        B, T, V = logits.shape
        device = logits.device

        # Softmax for backward use
        probs = F.softmax(logits, dim=-1)
        ctx.save_for_backward(probs, ref_ids)
        ctx.pad_id = pad_id
        ctx.eos_id = eos_id
        ctx.invert = invert

        # Hard argmax for forward BLEU
        pred_tokens = logits.argmax(dim=-1)  # (B, T)

        bleus = []
        ctx.pred_tokens = []
        ctx.ref_lists = []

        for b in range(B):
            ref_list = []
            for r in ref_ids[b]:
                r = r.item()
                if r in (pad_id, eos_id):
                    break
                ref_list.append(r)

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
        invert = getattr(ctx, 'invert', False)
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

            # Multi-head gradient: Token head (exact) + POS head (category)
            ref_tensor = torch.tensor(list(ref_set), device=device, dtype=torch.long)
            ref_freq_weight = ref_tensor.float() / V
            
            # Head 1: Token match (narrow, precise)
            token_match = (probs[b, :, ref_tensor] * (1.0 + ref_freq_weight * 10.0)).sum(dim=-1)
            token_signal = torch.exp(token_match * 20.0)
            
            # Head 2: POS category match (wide, ~200 categories)
            POS_BOUNDARY = 200
            ref_has_func = (ref_tensor < POS_BOUNDARY).any().float()
            ref_has_content = (ref_tensor >= POS_BOUNDARY).any().float()
            hyp_func = probs[b, :, :POS_BOUNDARY].sum(dim=-1)
            hyp_content = 1.0 - hyp_func
            pos_match = ref_has_func * hyp_func + ref_has_content * hyp_content
            pos_signal = torch.exp(pos_match * 10.0)
            
            # Multi-head combination: both signals contribute independently
            signal = token_signal + pos_signal
            
            if invert:
                signal = 1.0 / (signal + 1e-8)

            grad_raw = signal.unsqueeze(-1) * probs[b]
            grad_raw = grad_raw - grad_raw.mean(dim=-1, keepdim=True)
            grad_logits[b] = grad_raw * grad_output[b]

        return grad_logits, None, None, None, None, None


def logits_argmax(logits):
    """Helper for argmax on a single sample."""
    return logits.argmax(dim=-1)


def bleu_loss(logits, ref_ids, pad_id=0, eos_id=2, max_n=4, invert=False):
    """
    BLEU Function loss: L = 1 - BLEU.
    invert=True tests the inverted hash hypothesis.
    """
    bleu = BLEUFunction.apply(logits, ref_ids, pad_id, eos_id, max_n, invert)
    return (1.0 - bleu).mean(), bleu.mean().detach()


def bleu_ce_loss(logits, ref_ids, pad_id=0, eos_id=2, ce_weight=0.7, invert=False):
    """
    Hybrid: CE + BLEU Function. CE provides initial gradient,
    BLEU Function provides directional correction.
    """
    import torch.nn.functional as F
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ref_ids.reshape(-1), ignore_index=pad_id)
    bleu = BLEUFunction.apply(logits, ref_ids, pad_id, eos_id, 4, invert)
    bleu_loss_val = (1.0 - bleu).mean()
    total = ce_weight * ce + (1.0 - ce_weight) * bleu_loss_val
    return total, ce.detach(), bleu.mean().detach()
