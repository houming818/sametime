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


def train_epoch(model, loader, criterion, optimizer, soft_bleu_w=0.0, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        logits = model(src, tgt[:, :-1], src_len)

        if soft_bleu_w > 0:
            from base.soft_bleu import soft_bleu_loss
            ce_weight = 1.0 - soft_bleu_w
            loss, _, _ = soft_bleu_loss(logits, tgt[:, 1:], Vocab.PAD, Vocab.EOS, max_n=4, ce_weight=ce_weight)
        else:
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
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
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[Phase 2] device={DEVICE}")

    train_raw = load_iwslt14("train")
    vocab_src = Vocab([p["de"] for p in train_raw], min_freq=2)
    vocab_tgt = Vocab([p["en"] for p in train_raw], min_freq=2)
    print(f"  |src|={len(vocab_src)}  |tgt|={len(vocab_tgt)}")

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=64, shuffle=False)

    from model import Encoder, AttnDecoder
    encoder = Encoder(len(vocab_src), args.embed, args.hidden)
    decoder = AttnDecoder(len(vocab_tgt), args.embed, args.hidden)
    model = AttnSeq2Seq(encoder, decoder).to(DEVICE)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    start_epoch = 0
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        print(f"  resumed from epoch {ckpt['epoch']} (BLEU={ckpt['bleu']:.2f})")

    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, args.soft_bleu)
        bleu = evaluate(model, valid_loader, vocab_tgt)
        print(f"  epoch={epoch}  loss={loss:.3f}  BLEU={bleu:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint("checkpoints/phase2.pt", model, optimizer, epoch, bleu, {"phase": 2})
    print("[Phase 2] done")


if __name__ == "__main__":
    main()
