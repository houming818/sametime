"""
Phase 6: Transformer 训练入口
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from base.dataset import Vocab, load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics
from model import Transformer, Encoder, Decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WarmupScheduler:
    """学习率先线性升到 peak_lr，再指数衰减"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup ** -1.5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


def greedy_decode(model, src, src_len, vocab_tgt, max_len=50):
    model.eval()
    with torch.no_grad():
        enc_out = model.encoder(src, src_len)
        ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=DEVICE)
        for _ in range(max_len):
            logits = model.decoder(ys, enc_out, src_len)
            pred = logits[:, -1:, :].argmax(-1)
            ys = torch.cat([ys, pred], dim=1)
            if (pred == vocab_tgt.EOS).all():
                break
    return ys


def main():
    set_seed(42)
    print(f"[Phase 6 Transformer] device={DEVICE}")

    # ---- data ----
    train_raw = load_iwslt14("train")
    vocab_src = Vocab([p["de"] for p in train_raw], min_freq=2)
    vocab_tgt = Vocab([p["en"] for p in train_raw], min_freq=2)
    print(f"  |src|={len(vocab_src)}  |tgt|={len(vocab_tgt)}")

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=64, shuffle=False)

    # ---- Transformer 小配置 ----
    D_MODEL = 256
    N_LAYERS = 3
    N_HEADS = 4
    D_FF = 1024

    encoder = Encoder(len(vocab_src), D_MODEL, N_LAYERS, N_HEADS, D_FF)
    decoder = Decoder(len(vocab_tgt), D_MODEL, N_LAYERS, N_HEADS, D_FF)
    model = Transformer(encoder, decoder).to(DEVICE)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD, label_smoothing=0.1)

    # ---- train ----
    global_step = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt, src_len, tgt_len in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1], src_len)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()

        # evaluate
        model.eval()
        refs, hyps = [], []
        for src, tgt, src_len, tgt_len in valid_loader:
            src = src.to(DEVICE)
            out_ids = greedy_decode(model, src, src_len, vocab_tgt)
            for i in range(src.size(0)):
                refs.append(vocab_tgt.decode(tgt[i].tolist()))
                hyps.append(vocab_tgt.decode(out_ids[i].tolist()))
        bleu = compute_bleu(refs, hyps)
        avg_loss = total_loss / len(train_loader)
        print(f"  epoch={epoch}  loss={avg_loss:.3f}  BLEU={bleu:.2f}")
        log_metrics(epoch, avg_loss, bleu, optimizer.param_groups[0]["lr"])

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint("checkpoints/phase6.pt", model, optimizer, epoch, bleu, {"phase": 6})
    print("[Phase 6] done")


if __name__ == "__main__":
    main()
