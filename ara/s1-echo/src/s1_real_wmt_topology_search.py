"""Real WMT next-token topology search smoke.

Uses real WMT parallel-text English sentences.  This is a small language
model probe, not a translation benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")
TOPOLOGIES = {"direct": (0, 1, 2, 3), "balanced": ((0, 1), (2, 3)), "left_deep": (((0, 1), 2), 3), "right_deep": (0, (1, (2, 3))), "skip": ((0, 1), 2, 3)}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def internal_count(node) -> int:
    return 0 if isinstance(node, int) else 1 + sum(internal_count(x) for x in node)


def read_examples(path: Path, scan_lines: int, vocab_size: int, max_examples: int, seed: int):
    sentences = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle):
            if scan_lines > 0 and line_no >= scan_lines:
                break
            if "\t" not in line:
                continue
            left, right = line.rstrip("\n").split("\t", 1)
            left_cjk = sum("\u4e00" <= c <= "\u9fff" for c in left)
            right_cjk = sum("\u4e00" <= c <= "\u9fff" for c in right)
            text = right if left_cjk > right_cjk else left
            tokens = [t.lower() for t in TOKEN_RE.findall(text)]
            if len(tokens) >= 5:
                sentences.append(tokens)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    split = max(1, int(len(sentences) * 0.8))
    train_sentences, test_sentences = sentences[:split], sentences[split:]
    counter = Counter(t for s in train_sentences for t in s)
    vocab = ["<pad>", "<unk>"] + [t for t, _ in counter.most_common(vocab_size - 2)]
    stoi = {t: i for i, t in enumerate(vocab)}

    def make(sentences):
        rows = []
        for s in sentences:
            ids = [stoi.get(t, 1) for t in s]
            for i in range(len(ids) - 4):
                rows.append((ids[i : i + 4], ids[i + 4]))
                if max_examples > 0 and len(rows) >= max_examples:
                    return rows
        return rows

    return make(train_sentences), make(test_sentences), {"sentences": len(sentences), "train_sentences": len(train_sentences), "vocab": len(vocab), "vocab_preview": vocab[:20]}


class TopologyLM(nn.Module):
    def __init__(self, topology, vocab: int, dim: int):
        super().__init__()
        self.topology = topology
        self.embedding = nn.Embedding(vocab, dim)
        self.transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(internal_count(topology))])
        self.decoder = nn.Linear(dim, vocab)

    def fold(self, node, leaves, index):
        if isinstance(node, int):
            return leaves[:, node], index
        parts = []
        for child in node:
            value, index = self.fold(child, leaves, index)
            parts.append(value)
        base = torch.stack(parts, dim=1).mean(1)
        out = base + 0.5 * torch.tanh(self.transforms[index](base))
        return F.normalize(out, dim=-1), index + 1

    def forward(self, tokens):
        leaves = self.embedding(tokens)
        parent, _ = self.fold(self.topology, leaves, 0)
        return self.decoder(parent)


def evaluate(name, train, test, vocab, steps, seed, device, batch_size):
    seed_all(seed)
    model = TopologyLM(TOPOLOGIES[name], vocab, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.tensor([r[0] for r in train], dtype=torch.long, device=device)
    y = torch.tensor([r[1] for r in train], dtype=torch.long, device=device)
    xt = torch.tensor([r[0] for r in test], dtype=torch.long, device=device)
    yt = torch.tensor([r[1] for r in test], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 55)
    for _ in range(steps):
        idx = torch.randint(len(x), (min(batch_size, len(x)),), generator=gen)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        logits = model(xt)
        nll = float(F.cross_entropy(logits, yt).item())
        acc = float((logits.argmax(-1) == yt).float().mean().item())
    return {"topology": name, "test_nll": nll, "test_ppl": math.exp(min(nll, 20)), "test_accuracy": acc, "complexity": internal_count(TOPOLOGIES[name]), "finite": bool(torch.isfinite(logits).all().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scan-lines", type=int, default=50000)
    ap.add_argument("--max-examples", type=int, default=12000)
    ap.add_argument("--test-examples", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train, test, meta = read_examples(Path(args.data), args.scan_lines, args.vocab, args.max_examples, args.seed)
    test = test[: args.test_examples]
    if len(train) < 100 or len(test) < 100:
        raise RuntimeError(f"insufficient examples train={len(train)} test={len(test)}")
    rows = []
    current, current_score, best = "balanced", None, None
    rng = random.Random(args.seed + 7)
    temperature = 0.10
    for it in range(args.iterations):
        proposal = rng.choice(list(TOPOLOGIES))
        result = evaluate(proposal, train, test, meta["vocab"], args.steps, args.seed + 100 + it, device, 256)
        score = -result["test_nll"] - 0.01 * result["complexity"]
        accepted = current_score is None or score >= current_score or rng.random() < math.exp((score - current_score) / temperature)
        if accepted:
            current, current_score = proposal, score
        if best is None or score > best["score"]:
            best = {"topology": proposal, "score": score, **{k: result[k] for k in ("test_nll", "test_ppl", "test_accuracy", "complexity")}, "iteration": it}
        result.update({"iteration": it, "proposal": proposal, "accepted": accepted, "score": score, "current": current, "current_score": current_score, "temperature": temperature})
        rows.append(result)
        temperature *= 0.92
        print(json.dumps(result), flush=True)
    summary = {"seed": args.seed, "device": str(device), "data": args.data, "data_meta": meta, "train_examples": len(train), "test_examples": len(test), "steps_per_candidate": args.steps, "iterations": args.iterations, "global_best": best, "final_current": current, "final_score": current_score, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global_best": best, "final_current": current}, indent=2), flush=True)


if __name__ == "__main__":
    main()
