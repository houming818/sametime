"""
Phase 2: Bahdanau Attention 训练入口 （SoftBLEU 支持）
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from base.dataset import Vocab, load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics, load_checkpoint
from model import AttnSeq2Seq

def _log(**kw):
    """Print logfmt line for Loki parsing."""
    parts = []
    for k, v in kw.items():
        if isinstance(v, float):
            parts.append(f'{k}={v:.4f}')
        elif isinstance(v, bool):
            parts.append(f'{k}={str(v).lower()}')
        else:
            parts.append(f'{k}={v}')
    print(' '.join(parts), flush=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(model, src, src_len, vocab_tgt, max_len=50):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src, src_len)
        ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=DEVICE)
        for _ in range(max_len):
            logits, hidden = model.decoder(ys, (enc_out, hidden))
            pred = logits[:, -1:, :].argmax(-1)
            ys = torch.cat([ys, pred], dim=1)
            if (pred == vocab_tgt.EOS).all(): break
    return ys


def train_epoch(model, loader, criterion, optimizer, soft_bleu_w=0.0, dual_head=False,
                 restricted_bleu=False, k_lang=170, sb_mode='multiply_linear',
                 sb_alpha=0.5, sb_beta=0.5, exp_name='', clip=1.0):
    model.train()
    total_loss = 0
    batch_count = 0
    eps = 1e-8
    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        batch_count += 1
        
        if dual_head:
            ce_logits, sb_logits = model(src, tgt[:, :-1], src_len)
            ce_loss = criterion(ce_logits.reshape(-1, ce_logits.size(-1)), tgt[:, 1:].reshape(-1))
            if restricted_bleu:
                from base.soft_bleu import soft_bleu_restricted
                sb_bleu_val, _ = soft_bleu_restricted(sb_logits, tgt[:, 1:], 0, 2, 4, k=k_lang)
            else:
                from base.soft_bleu import soft_bleu
                sb_bleu_val, _ = soft_bleu(sb_logits, tgt[:, 1:], 0, 2, 4)

            sb_loss = (1.0 - sb_bleu_val).clamp(min=eps)
            if sb_mode == 'multiply_sqrt':
                # f(SB) = (1−SB)^β, f'(SB) = −β/(1−SB)^(1−β)
                # gradient auto-amplifies by 1/(1−SB)^(1−β) as SB plateaus
                sb_term = sb_loss.pow(sb_beta)
            else:
                # multiply_linear: f(SB) = 1−SB, constant gradient decay
                sb_term = sb_loss
            loss = ce_loss * (1.0 + sb_alpha * sb_term)

        elif soft_bleu_w > 0:
            logits = model(src, tgt[:, :-1], src_len)
            from base.soft_bleu import soft_bleu_loss, soft_bleu_only_loss
            if soft_bleu_w >= 1.0:
                loss, _, _ = soft_bleu_only_loss(logits, tgt[:, 1:], Vocab.PAD, Vocab.EOS)
            else:
                loss, _, _ = soft_bleu_loss(logits, tgt[:, 1:], Vocab.PAD, Vocab.EOS, max_n=4, ce_weight=1.0-soft_bleu_w)
        else:
            logits = model(src, tgt[:, :-1], src_len)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        if dual_head and batch_count % 50 == 0:
            grad_sb = model.decoder.out_sb.weight.grad.norm().item()
            grad_ce = model.decoder.out.weight.grad.norm().item()
            _log(exp=exp_name, batch=batch_count,
                 ce=ce_loss.item(), sb_bleu=sb_bleu_val.item(),
                 grad_sb=grad_sb, grad_ce=grad_ce, sb_alpha=sb_alpha)
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, vocab_tgt):
    model.eval()
    refs, hyps = [], []
    for src, tgt, src_len, tgt_len in loader:
        src = src.to(DEVICE)
        out_ids = greedy_decode(model, src, src_len, vocab_tgt)
        for i in range(src.size(0)):
            refs.append(vocab_tgt.decode(tgt[i].tolist()))
            hyps.append(vocab_tgt.decode(out_ids[i].tolist()))
    return compute_bleu(refs, hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft-bleu", type=float, default=0.0)
    parser.add_argument("--dual-head", action="store_true")
    parser.add_argument("--restricted-bleu", action="store_true")
    parser.add_argument("--k-lang", type=int, default=170, help="topk for restricted softmax")
    parser.add_argument("--sb-mode", type=str, default="multiply_linear",
                        choices=["multiply_linear", "multiply_sqrt"],
                        help="SB loss mode: linear=1-SB, sqrt=(1-SB)^beta auto-amplify")
    parser.add_argument("--sb-beta", type=float, default=0.5, help="sqrt decay exponent (0<b<1, smaller=stronger amplification)")
    parser.add_argument("--sb-alpha", type=float, default=0.5, help="SB loss weight")
    parser.add_argument("--sb-alpha-step", type=float, default=0.0, help="alpha increment per epoch")
    parser.add_argument("--output", type=str, default="checkpoints/phase2.pt")
    parser.add_argument("--exp-name", type=str, default="", help="logfmt experiment id")
    parser.add_argument("--exp-name-zh", type=str, default="", help="Chinese experiment name for logs")
    args = parser.parse_args()

    set_seed(args.seed)
    _log(exp=args.exp_name, exp_zh=args.exp_name_zh, phase=2,
         hid=args.hidden, emb=args.embed, ep=args.epochs, seed=args.seed,
         dual_head=args.dual_head, restricted=args.restricted_bleu,
         k_lang=args.k_lang, sb_mode=args.sb_mode,
         sb_alpha=args.sb_alpha, sb_beta=args.sb_beta,
         device=str(DEVICE))

    train_raw = load_iwslt14("train")
    vocab_src = Vocab([p["de"] for p in train_raw], min_freq=2)
    vocab_tgt = Vocab([p["en"] for p in train_raw], min_freq=2)
    _log(exp=args.exp_name, vsrc=len(vocab_src), vtgt=len(vocab_tgt))

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=64, shuffle=False)

    from model import Encoder, AttnDecoder
    encoder = Encoder(len(vocab_src), args.embed, args.hidden)
    decoder = AttnDecoder(len(vocab_tgt), args.embed, args.hidden, dual_head=args.dual_head)
    model = AttnSeq2Seq(encoder, decoder).to(DEVICE)
    _log(exp=args.exp_name, params=sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    start_epoch = 0
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        print(f"  resumed from epoch {ckpt['epoch']} (BLEU={ckpt['bleu']:.2f})")

    for epoch in range(start_epoch, args.epochs):
        sb_alpha_eff = args.sb_alpha + epoch * args.sb_alpha_step
        if args.dual_head:
            sb_norm_before = model.decoder.out_sb.weight.data.norm().item()
        loss = train_epoch(model, train_loader, criterion, optimizer, args.soft_bleu,
                          dual_head=args.dual_head, restricted_bleu=args.restricted_bleu,
                          k_lang=args.k_lang, sb_mode=args.sb_mode,
                          sb_alpha=sb_alpha_eff, sb_beta=args.sb_beta,
                          exp_name=args.exp_name)
        bleu = evaluate(model, valid_loader, vocab_tgt)
        log_kw = dict(exp=args.exp_name, epoch=epoch, train_loss=loss, bleu=bleu)
        if args.dual_head:
            sb_norm_after = model.decoder.out_sb.weight.data.norm().item()
            log_kw.update(sb_norm_before=sb_norm_before, sb_norm_after=sb_norm_after,
                         sb_delta=sb_norm_after - sb_norm_before)
        _log(**log_kw)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_checkpoint(args.output, model, optimizer, epoch, bleu, {"phase": 2})
    _log(exp=args.exp_name, status='done', final_bleu=bleu)


if __name__ == "__main__":
    main()
