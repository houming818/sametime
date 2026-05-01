"""
Phase 1.0: Vanilla RNN Seq2Seq 训练入口
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
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(model, src, src_len, vocab_tgt, max_len=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src, src_len)
        ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=device)
        for _ in range(max_len):
            logits, hidden = model.decoder(ys, hidden)
            pred = logits[:, -1:, :].argmax(-1)
            ys = torch.cat([ys, pred], dim=1)
            if (pred == vocab_tgt.EOS).all():
                break
    return ys


def train_epoch(model, loader, criterion, optimizer, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        logits = model(src, tgt[:, :-1], src_len)
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         tgt[:, 1:].reshape(-1))
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
        out_ids = greedy_decode(model, src, src_len, vocab_tgt, device=DEVICE)
        for i in range(src.size(0)):
            refs.append(vocab_tgt.decode(tgt[i].tolist()))
            hyps.append(vocab_tgt.decode(out_ids[i].tolist()))
    return compute_bleu(refs, hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--enc-embed", type=int, default=None)
    parser.add_argument("--dec-embed", type=int, default=None)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[Phase 1.0 RNN] device={DEVICE}")

    # ---- data ----
    print("  building vocab...")
    train_raw = load_iwslt14("train")
    vocab_src = Vocab([p["de"] for p in train_raw], min_freq=2)
    vocab_tgt = Vocab([p["en"] for p in train_raw], min_freq=2)
    print(f"  |src|={len(vocab_src)}  |tgt|={len(vocab_tgt)}")

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=args.batch_size)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=args.batch_size, shuffle=False)

    # ---- model ----
    enc_embed = args.enc_embed if args.enc_embed is not None else args.embed
    dec_embed = args.dec_embed if args.dec_embed is not None else args.embed
    encoder = Encoder(len(vocab_src), enc_embed, args.hidden, args.layers, args.dropout)
    decoder = Decoder(len(vocab_tgt), dec_embed, args.hidden, args.layers, args.dropout)
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    start_epoch = 0

    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        print(f"  resumed from epoch {ckpt['epoch']} (BLEU={ckpt['bleu']:.2f})")

    # ---- train ----
    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        bleu = evaluate(model, valid_loader, vocab_tgt)
        print(f"  epoch={epoch}  loss={loss:.3f}  BLEU={bleu:.2f}")
        log_metrics(epoch, loss, bleu, args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint("checkpoints/phase1_0_rnn.pt", model, optimizer, epoch, bleu, {"phase": "1.0"})
    print("[Phase 1.0 RNN] done")


if __name__ == "__main__":
    main()
