"""
Phase 3: Luong Attention + Beam Search 训练入口
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from base.dataset import Vocab, load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics, load_checkpoint
from model import Encoder, LuongDecoder, LuongSeq2Seq
from beam_search import beam_search

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, vocab_tgt, beam_size=3):
    model.eval()
    refs, hyps = [], []
    for src, tgt, src_len, tgt_len in loader:
        src = src.to(DEVICE)
        for i in range(src.size(0)):
            src_i = src[i:i+1]
            src_len_i = src_len[i:i+1]
            out_ids = beam_search(model, src_i, src_len_i, vocab_tgt, beam_size=beam_size)
            refs.append(vocab_tgt.decode(tgt[i].tolist()))
            hyps.append(vocab_tgt.decode(out_ids[0].tolist()))
    return compute_bleu(refs, hyps)


def train_epoch(model, loader, criterion, optimizer, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        logits = model(src, tgt[:, :-1], src_len)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--beam", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    print(f"[Phase 3] device={DEVICE}  beam_size={args.beam}")

    train_raw = load_iwslt14("train")
    vocab_src = Vocab([p["de"] for p in train_raw], min_freq=2)
    vocab_tgt = Vocab([p["en"] for p in train_raw], min_freq=2)
    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=1, shuffle=False)

    HIDDEN, EMBED = 256, 256
    encoder = Encoder(len(vocab_src), EMBED, HIDDEN)
    decoder = LuongDecoder(len(vocab_tgt), EMBED, HIDDEN, attn_method="general")
    model = LuongSeq2Seq(encoder, decoder).to(DEVICE)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD)
    start_epoch = 0
    if args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        bleu = evaluate(model, valid_loader, vocab_tgt, args.beam)
        print(f"  epoch={epoch}  loss={loss:.3f}  BLEU={bleu:.2f}")
        log_metrics(epoch, loss, bleu, 1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint("checkpoints/phase3.pt", model, optimizer, epoch, bleu, {"phase": 3})
    print("[Phase 3] done")


if __name__ == "__main__":
    main()
