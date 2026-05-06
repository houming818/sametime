"""
Phase 6 BPE: Transformer + SentencePiece BPE tokenization
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentencepiece as spm
import torch, torch.optim as optim, torch.nn.functional as F

from base.dataset import load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics
from model import Transformer, Encoder, Decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPEVocab:
    SOS, EOS, PAD = 1, 2, 0
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    def __len__(self): return self.sp.vocab_size()
    def encode(self, s, max_len=50):
        ids = self.sp.encode(s, out_type=int)[:max_len-2]
        return [self.SOS] + ids + [self.EOS]
    def decode(self, ids):
        ids = [i for i in ids if i not in (self.PAD, self.SOS, self.EOS)]
        return self.sp.decode(ids)


class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer; self.d_model = d_model
        self.warmup = warmup_steps; self.step_num = 0
    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * self.warmup ** -1.5)
        for pg in self.optimizer.param_groups: pg["lr"] = lr


def greedy_decode(model, src, src_len, vocab_tgt, max_len=50):
    model.eval()
    with torch.no_grad():
        enc_out, src_mask = model.encoder(src, src_len)
        ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=DEVICE)
        for _ in range(max_len):
            logits = model.decoder(ys, enc_out, src_mask)
            if isinstance(logits, tuple):
                logits = logits[0]  # use CE head for decode
            pred = logits[:, -1:, :].argmax(-1)
            ys = torch.cat([ys, pred], dim=1)
            if (pred == vocab_tgt.EOS).all(): break
    return ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--output", type=str, default="checkpoints/phase6_bpe.pt")
    parser.add_argument("--exp-name", type=str, default="B?")
    parser.add_argument("--dual-head", action="store_true")
    parser.add_argument("--restricted-bleu", action="store_true")
    parser.add_argument("--k-lang", type=int, default=170)
    parser.add_argument("--sb-alpha", type=float, default=0.8)
    args = parser.parse_args()

    set_seed(42)
    print(f"exp={args.exp_name} phase=6_bpe device=cuda d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} d_ff={args.d_ff} epochs={args.epochs}")

    vocab_src = BPEVocab("checkpoints/spm_de.model")
    vocab_tgt = BPEVocab("checkpoints/spm_en.model")
    print(f"exp={args.exp_name} vsrc={len(vocab_src)} vtgt={len(vocab_tgt)}")

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=args.batch_size)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder(len(vocab_src), args.d_model, args.n_layers, args.n_heads, args.d_ff)
    decoder = Decoder(len(vocab_tgt), args.d_model, args.n_layers, args.n_heads, args.d_ff, dual_head=args.dual_head)
    model = Transformer(encoder, decoder).to(DEVICE)
    print(f"exp={args.exp_name} params={sum(p.numel() for p in model.parameters()):,}" + 
          (f" dual_head=true k={args.k_lang} alpha={args.sb_alpha}" if args.dual_head else ""))

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, args.d_model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=BPEVocab.PAD, label_smoothing=0.1)

    best_bleu = 0
    for epoch in range(args.epochs):
        model.train(); total_loss = 0
        for src, tgt, src_len, tgt_len in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            if args.dual_head:
                ce_logits, sb_logits = model(src, tgt[:, :-1], src_len)
                ce_loss = criterion(ce_logits.reshape(-1, ce_logits.size(-1)), tgt[:, 1:].reshape(-1))
                from base.soft_bleu import soft_bleu_restricted
                sb_bleu_val, _ = soft_bleu_restricted(sb_logits, tgt[:, 1:], 0, 2, 4, k=args.k_lang)
                loss = ce_loss * (1.0 + args.sb_alpha * (1.0 - sb_bleu_val))
            else:
                logits = model(src, tgt[:, :-1], src_len)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

        bleu = 0
        if epoch >= 5 or epoch == 0:
            model.eval(); refs, hyps = [], []
            for src, tgt, src_len, tgt_len in valid_loader:
                src = src.to(DEVICE)
                out_ids = greedy_decode(model, src, src_len, vocab_tgt)
                for i in range(src.size(0)):
                    refs.append(vocab_tgt.decode(tgt[i].tolist()))
                    hyps.append(vocab_tgt.decode(out_ids[i].tolist()))
            bleu = compute_bleu(refs, hyps)
        avg_loss = total_loss / len(train_loader)
        print(f"exp={args.exp_name} epoch={epoch} loss={avg_loss:.3f} bleu={bleu:.2f}")
        log_metrics(epoch, avg_loss, bleu, optimizer.param_groups[0]["lr"])
        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            save_checkpoint(args.output, model, optimizer, epoch, bleu, {"phase": "6_bpe"})

    print(f"exp={args.exp_name} status=done best_bleu={best_bleu:.2f}")


if __name__ == "__main__":
    main()
