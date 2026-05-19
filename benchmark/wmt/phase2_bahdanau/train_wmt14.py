"""
AW — Bahdanau Attention on WMT14 De-En (BPE tokenization)
A.STD.CE-036: d256/3L/30ep
A.STD.CE-037: d512/3L/30ep
"""
import sys, os, argparse
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sentencepiece as spm
import torch, torch.optim as optim

from base.dataset import load_wmt14_de_en, build_dataloader_wmt14, Vocab
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint
from model import Encoder, AttnDecoder, AttnSeq2Seq

PAD = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPEVocab:
    SOS, EOS, PAD = 1, 2, 0
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    def __len__(self):
        return self.sp.vocab_size()
    def encode(self, s, max_len=128):
        ids = self.sp.encode(s, out_type=int)[:max_len-2]
        return [self.SOS] + ids + [self.EOS]
    def decode(self, ids):
        ids = [i for i in ids if i not in (self.PAD, self.SOS, self.EOS)]
        return self.sp.decode(ids)


def greedy_decode(model, src, src_len, vocab_tgt, max_len=128):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src, src_len)
        ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=DEVICE)
        for _ in range(max_len):
            logits, hidden = model.decoder(ys, (enc_out, hidden))
            pred = logits[:, -1:, :].argmax(-1)
            ys = torch.cat([ys, pred], dim=1)
            if (pred == vocab_tgt.EOS).all():
                break
    return ys


def train_epoch(model, loader, criterion, optimizer, scaler, clip=1.0):
    model.train()
    total_loss = 0
    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        with torch.cuda.amp.autocast():
            logits = model(src, tgt[:, :-1], src_len)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256, help="embed_size = hidden_size")
    parser.add_argument("--n-layers", type=int, default=3, help="LSTM layers (encoder+decoder)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="checkpoints/aw.pt")
    parser.add_argument("--exp-name", type=str, default="AW?")
    args = parser.parse_args()

    set_seed(42)
    print(f"exp={args.exp_name} phase=2_wmt14 device=cuda d_model={args.d_model} lstm_layers={args.n_layers} epochs={args.epochs}")

    vocab_src = BPEVocab("checkpoints/wmt14_spm_de.model")
    vocab_tgt = BPEVocab("checkpoints/wmt14_spm_en.model")
    print(f"exp={args.exp_name} vsrc={len(vocab_src)} vtgt={len(vocab_tgt)}")

    train_loader = build_dataloader_wmt14("train", vocab_src, vocab_tgt, batch_size=args.batch_size)
    valid_loader = build_dataloader_wmt14("validation", vocab_src, vocab_tgt, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder(len(vocab_src), args.d_model, args.d_model, num_layers=args.n_layers, dropout=args.dropout)
    decoder = AttnDecoder(len(vocab_tgt), args.d_model, args.d_model, num_layers=args.n_layers, dropout=args.dropout)
    model = AttnSeq2Seq(encoder, decoder).to(DEVICE)
    torch.backends.cudnn.benchmark = True
    print(f"exp={args.exp_name} params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    scaler = torch.cuda.amp.GradScaler()

    best_bleu = 0
    for epoch in range(args.epochs):
        t0 = datetime.now()
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        bleu = evaluate(model, valid_loader, vocab_tgt) if epoch >= 5 or epoch == 0 else 0
        elapsed = (datetime.now() - t0).total_seconds()
        now = datetime.now().strftime("%m-%d_%H:%M")
        print(f"[{now}] exp={args.exp_name} epoch={epoch} loss={loss:.3f} bleu={bleu:.2f} time={elapsed:.0f}s")
        if bleu > best_bleu:
            best_bleu = bleu
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            save_checkpoint(args.output, model, optimizer, epoch, bleu, {"phase": "2_wmt14"})

    print(f"exp={args.exp_name} status=done best_bleu={best_bleu:.2f}")


if __name__ == "__main__":
    main()
