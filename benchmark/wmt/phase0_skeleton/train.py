"""
Phase 0: 实验底座骨架

用最简"复制模型"验证数据 + 评测 + 训练循环全部跑通。
这个模型不做翻译——只是把 src 原样当 tgt 输出（BLEU 会低得离谱，没关系）。
目的：确保 dataset / eval / checkpoint 全部可运行。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim

from base.dataset import Vocab, load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics


class DummyModel(nn.Module):
    def __init__(self, vocab_tgt_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_tgt_size, 64, padding_idx=Vocab.PAD)
        self.out = nn.Linear(64, vocab_tgt_size)

    def forward(self, src, tgt_in):
        x = self.embed(tgt_in)
        return self.out(x)  # (B, T-1, V)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 0] device={device}")

    # ---- 1. 建立词表 ----
    print("  building vocab...")
    train_raw = load_iwslt14("train")
    src_sentences = [p["de"] for p in train_raw]
    tgt_sentences = [p["en"] for p in train_raw]
    vocab_src = Vocab(src_sentences, min_freq=2)
    vocab_tgt = Vocab(tgt_sentences, min_freq=2)
    print(f"  |vocab_src|={len(vocab_src)}  |vocab_tgt|={len(vocab_tgt)}")

    # ---- 2. DataLoader ----
    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=64, shuffle=False)

    # ---- 3. 模型 ----
    model = DummyModel(len(vocab_tgt)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.PAD)

    # ---- 4. 训练 ----
    for epoch in range(2):
        model.train()
        total_loss = 0
        for src, tgt, src_len, tgt_len in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])  # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- 5. 验证 ----
        model.eval()
        refs, hyps = [], []
        for src, tgt, src_len, tgt_len in valid_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            preds = logits.argmax(-1)
            for i in range(src.size(0)):
                refs.append(vocab_tgt.decode(tgt[i].tolist()))
                hyps.append(vocab_tgt.decode(preds[i].tolist()))
        bleu = compute_bleu(refs, hyps)
        avg_loss = total_loss / len(train_loader)

        print(f"  epoch={epoch}  loss={avg_loss:.3f}  BLEU={bleu:.2f}")
        log_metrics(epoch, avg_loss, bleu, 1e-3)

    save_checkpoint("checkpoints/phase0.pt", model, optimizer, epoch, bleu, {"phase": 0})
    print("[Phase 0] done")


if __name__ == "__main__":
    main()
