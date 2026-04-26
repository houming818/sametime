"""
Phase 4: BPE 版本训练入口

用 SentencePiece 替换 word-level Vocab，其他代码与 Phase 2 相同。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentencepiece as spm
import torch
import torch.optim as optim

from base.dataset import load_iwslt14, build_dataloader
from base.eval import compute_bleu
from base.utils import set_seed, save_checkpoint, log_metrics

# 复用 Phase 2 的 model
from phase2_bahdanau.model import Encoder, AttnDecoder, AttnSeq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPEVocab:
    """BPE 封装，兼容 Vocab 接口"""
    SOS = 1
    EOS = 2
    PAD = 0

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def __len__(self):
        return self.sp.vocab_size()

    def encode(self, sentence, max_len=50):
        ids = self.sp.encode(sentence, out_type=int)
        ids = ids[:max_len - 2]
        return [self.SOS] + ids + [self.EOS]

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.PAD, self.SOS, self.EOS)]
        return self.sp.decode(ids)


def main():
    set_seed(42)
    print(f"[Phase 4 BPE] device={DEVICE}")

    # ---- data with BPE ----
    vocab_src = BPEVocab("checkpoints/spm_de.model")
    vocab_tgt = BPEVocab("checkpoints/spm_en.model")
    print(f"  |src|={len(vocab_src)}  |tgt|={len(vocab_tgt)}")

    train_loader = build_dataloader("train", vocab_src, vocab_tgt, batch_size=64)
    valid_loader = build_dataloader("validation", vocab_src, vocab_tgt, batch_size=64, shuffle=False)

    # ---- model ----
    HIDDEN, EMBED = 256, 256
    encoder = Encoder(len(vocab_src), EMBED, HIDDEN)
    decoder = AttnDecoder(len(vocab_tgt), EMBED, HIDDEN)
    model = AttnSeq2Seq(encoder, decoder).to(DEVICE)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=Vocab.PAD)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for src, tgt, src_len, tgt_len in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt[:, :-1], src_len)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # eval
        model.eval()
        refs, hyps = [], []
        for src, tgt, src_len, tgt_len in valid_loader:
            src = src.to(DEVICE)
            enc_out, hidden = model.encoder(src, src_len)
            ys = torch.full((src.size(0), 1), vocab_tgt.SOS, dtype=torch.long, device=DEVICE)
            for _ in range(50):
                logits, hidden = model.decoder(ys, (enc_out, hidden))
                pred = logits[:, -1:, :].argmax(-1)
                ys = torch.cat([ys, pred], dim=1)
                if (pred == vocab_tgt.EOS).all():
                    break
            for i in range(src.size(0)):
                refs.append(vocab_tgt.decode(tgt[i].tolist()))
                hyps.append(vocab_tgt.decode(ys[i].tolist()))

        bleu = compute_bleu(refs, hyps)
        avg_loss = total_loss / len(train_loader)
        print(f"  epoch={epoch}  loss={avg_loss:.3f}  BLEU={bleu:.2f}")
        log_metrics(epoch, avg_loss, bleu, 1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint("checkpoints/phase4.pt", model, optimizer, epoch, bleu, {"phase": 4})
    print("[Phase 4 BPE] done")


if __name__ == "__main__":
    main()
