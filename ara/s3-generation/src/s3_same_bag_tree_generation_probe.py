#!/usr/bin/env python3
"""S3 same-bag different-tree generation proof.

This is the first generation gate after the frozen decoder bucket proof.

The task is intentionally controlled:

  same leaf sequence  [a, b, c]
  same bag of tokens  {a, b, c}
  different TreeHeap structure
  different generated output

Shape 0:
    ((a b) c)  ->  generate: PAIR a b

Shape 1:
    (a (b c))  ->  generate: PAIR b c

BoW and ordinary flat sequence baselines receive only [a, b, c].  Since every
triple appears with both shapes and two different targets, they cannot solve
the ambiguity from token identity/order alone.  TreeHeap receives the tree
structure and decodes from the internal subheap selected by that structure.

This does not prove WMT.  It tests the minimum S3 claim:

  generation must use TreeHeap substructure, not only token bag statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


PAIR = 0


@dataclass
class Config:
    evidence_dir: str
    seed: int
    vocab_size: int
    train_triples: int
    test_triples: int
    ood_triples: int
    dim: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    device: str
    models: List[str]
    report_every: int


def make_triples(vocab_size: int, seed: int) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    leaves = list(range(1, vocab_size))
    triples: List[Tuple[int, int, int]] = []
    for a in leaves:
        for b in leaves:
            if b == a:
                continue
            for c in leaves:
                if c == a or c == b:
                    continue
                triples.append((a, b, c))
    rng.shuffle(triples)
    return triples


def build_split(cfg: Config) -> Dict[str, torch.Tensor]:
    need = cfg.train_triples + cfg.test_triples + cfg.ood_triples
    triples = make_triples(cfg.vocab_size, cfg.seed)
    if len(triples) < need:
        raise ValueError(f"need {need} triples, only have {len(triples)}")

    train = triples[: cfg.train_triples]
    test = triples[cfg.train_triples : cfg.train_triples + cfg.test_triples]
    ood = triples[cfg.train_triples + cfg.test_triples : need]

    def expand(rows: Sequence[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        leaves: List[List[int]] = []
        shapes: List[int] = []
        targets: List[List[int]] = []
        for a, b, c in rows:
            leaves.append([a, b, c])
            shapes.append(0)
            targets.append([PAIR, a, b])

            leaves.append([a, b, c])
            shapes.append(1)
            targets.append([PAIR, b, c])
        return (
            torch.tensor(leaves, dtype=torch.long),
            torch.tensor(shapes, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    train_x, train_s, train_y = expand(train)
    test_x, test_s, test_y = expand(test)
    ood_x, ood_s, ood_y = expand(ood)
    return {
        "train_x": train_x,
        "train_s": train_s,
        "train_y": train_y,
        "test_x": test_x,
        "test_s": test_s,
        "test_y": test_y,
        "ood_x": ood_x,
        "ood_s": ood_s,
        "ood_y": ood_y,
    }


class BagGenerator(nn.Module):
    kind = "bow"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3 * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        h = self.emb(x).mean(dim=1)
        return self.mlp(h).view(x.shape[0], 3, self.vocab_size)


class FlatSeqGenerator(nn.Module):
    kind = "flat_seq"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(3, dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3 * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        return self.mlp(h.reshape(x.shape[0], -1)).view(x.shape[0], 3, self.vocab_size)


class ShapeOracleGenerator(nn.Module):
    kind = "shape_oracle"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.shape = nn.Embedding(2, dim)
        self.pos = nn.Embedding(3, dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3 * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        leaf = self.emb(x) + self.pos(pos)
        h = torch.cat([leaf.reshape(x.shape[0], -1), self.shape(shape)], dim=-1)
        return self.mlp(h).view(x.shape[0], 3, self.vocab_size)


class TreeHeapGenerator(nn.Module):
    kind = "treeheap"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.left_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.right_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3 * vocab_size),
        )
        self.vocab_size = vocab_size

    def plus(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Role offsets make left/right order non-commutative.
        pair = torch.cat([left + self.left_role, right + self.right_role], dim=-1)
        return self.compose(pair)

    def forward(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        a, b, c = self.emb(x[:, 0]), self.emb(x[:, 1]), self.emb(x[:, 2])
        ab = self.plus(a, b)
        bc = self.plus(b, c)

        # The useful internal subheap is created by the TreeHeap structure:
        #   shape 0: ((a b) c) stops at ab
        #   shape 1: (a (b c)) stops at bc
        stop = torch.where(shape.unsqueeze(-1) == 0, ab, bc)
        return self.decoder(stop).view(x.shape[0], 3, self.vocab_size)


MODEL_REGISTRY = {
    "bow": BagGenerator,
    "flat_seq": FlatSeqGenerator,
    "shape_oracle": ShapeOracleGenerator,
    "treeheap": TreeHeapGenerator,
}


def batch_indices(n: int, batch_size: int, rng: random.Random) -> Iterable[List[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def decode(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = decode(logits)
    token_acc = (pred == target).float().mean().item()
    exact = (pred == target).all(dim=1).float().mean().item()
    first_arg = (pred[:, 1] == target[:, 1]).float().mean().item()
    second_arg = (pred[:, 2] == target[:, 2]).float().mean().item()
    return {
        "token_acc": token_acc,
        "exact": exact,
        "first_arg_acc": first_arg,
        "second_arg_acc": second_arg,
        "loss": float(ce_loss(logits, target).item()),
    }


def contradiction_rate(x: torch.Tensor, y: torch.Tensor) -> float:
    by_x: Dict[Tuple[int, int, int], set[Tuple[int, int, int]]] = {}
    for leaves, target in zip(x.tolist(), y.tolist()):
        by_x.setdefault(tuple(leaves), set()).add(tuple(target))
    ambiguous = sum(1 for targets in by_x.values() if len(targets) > 1)
    return ambiguous / max(1, len(by_x))


def train_model(
    name: str,
    arrays: Dict[str, torch.Tensor],
    cfg: Config,
) -> Dict[str, object]:
    torch.manual_seed(cfg.seed + sum(ord(c) for c in name))
    rng = random.Random(cfg.seed + 1000 + sum(ord(c) for c in name))
    model = MODEL_REGISTRY[name](cfg.vocab_size, cfg.dim, cfg.hidden).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_x = arrays["train_x"].to(cfg.device)
    train_s = arrays["train_s"].to(cfg.device)
    train_y = arrays["train_y"].to(cfg.device)
    trace: List[Dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_n = 0
        model.train()
        for idx in batch_indices(train_x.shape[0], cfg.batch_size, rng):
            sel = torch.tensor(idx, dtype=torch.long, device=cfg.device)
            logits = model(train_x[sel], train_s[sel])
            loss = ce_loss(logits, train_y[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
            total_n += len(idx)
        if epoch == 1 or epoch == cfg.epochs or (cfg.report_every and epoch % cfg.report_every == 0):
            with torch.no_grad():
                train_m = metrics(model(train_x, train_s), train_y)
            row = {
                "epoch": epoch,
                "loss": total_loss / max(1, total_n),
                "train_exact": train_m["exact"],
                "train_token_acc": train_m["token_acc"],
            }
            trace.append(row)
            print(f"[{name}] epoch={epoch} loss={row['loss']:.4f} exact={row['train_exact']:.4f}", flush=True)

    model.eval()
    out: Dict[str, object] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trace": trace,
        "metrics": {},
        "examples": {},
    }
    with torch.no_grad():
        for split in ["train", "test", "ood"]:
            x = arrays[f"{split}_x"].to(cfg.device)
            s = arrays[f"{split}_s"].to(cfg.device)
            y = arrays[f"{split}_y"].to(cfg.device)
            logits = model(x, s)
            out["metrics"][split] = metrics(logits.cpu(), y.cpu())  # type: ignore[index]
            pred = decode(logits).cpu()
            examples = []
            for i in range(min(8, x.shape[0])):
                examples.append(
                    {
                        "leaves": x[i].detach().cpu().tolist(),
                        "shape": int(s[i].detach().cpu().item()),
                        "target": y[i].detach().cpu().tolist(),
                        "pred": pred[i].tolist(),
                        "exact": bool(torch.equal(pred[i], y[i].detach().cpu())),
                    }
                )
            out["examples"][split] = examples  # type: ignore[index]
    return out


def summarize(model_rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    compact: Dict[str, object] = {}
    for name, row in model_rows.items():
        compact[name] = {
            "parameters": row["parameters"],
            "ood": row["metrics"]["ood"],  # type: ignore[index]
            "test": row["metrics"]["test"],  # type: ignore[index]
        }
    if "treeheap" in compact and "bow" in compact and "flat_seq" in compact:
        tree = compact["treeheap"]["ood"]  # type: ignore[index]
        bow = compact["bow"]["ood"]  # type: ignore[index]
        flat = compact["flat_seq"]["ood"]  # type: ignore[index]
        compact["primary_gaps"] = {
            "treeheap_minus_bow_exact": tree["exact"] - bow["exact"],
            "treeheap_minus_flat_seq_exact": tree["exact"] - flat["exact"],
            "treeheap_minus_bow_token_acc": tree["token_acc"] - bow["token_acc"],
            "treeheap_minus_flat_seq_token_acc": tree["token_acc"] - flat["token_acc"],
        }
    return compact


def write_outputs(
    evidence_dir: Path,
    cfg: Config,
    arrays: Dict[str, torch.Tensor],
    model_rows: Dict[str, Dict[str, object]],
    started: float,
) -> Dict[str, object]:
    summary = {
        "claim": "S3-SAMEBAG-GEN-C01",
        "experiment": "P-S3-SAMEBAG-GEN01",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": asdict(cfg),
        "data": {
            "train_items": int(arrays["train_x"].shape[0]),
            "test_items": int(arrays["test_x"].shape[0]),
            "ood_items": int(arrays["ood_x"].shape[0]),
            "train_contradiction_rate_without_tree": contradiction_rate(arrays["train_x"], arrays["train_y"]),
            "test_contradiction_rate_without_tree": contradiction_rate(arrays["test_x"], arrays["test_y"]),
            "ood_contradiction_rate_without_tree": contradiction_rate(arrays["ood_x"], arrays["ood_y"]),
        },
        "models": model_rows,
        "summary": summarize(model_rows),
        "decision_hint": {
            "support_if": "treeheap beats bow and flat_seq on OOD exact/token generation when same token sequence has contradictory targets",
            "stronger_next_gate": "replace controlled symbols with real parsed spans or WMT canonical substructures",
            "not_proved": "WMT translation, natural syntax induction, or superiority over models that are also given the tree structure",
        },
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (evidence_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for model_name, row in model_rows.items():
            for trace_row in row["trace"]:  # type: ignore[index]
                f.write(json.dumps({"model": model_name, **trace_row}, ensure_ascii=False) + "\n")

    lines = [
        "# S3 Same-Bag Different-Tree Generation Probe",
        "",
        "The input leaf sequence is identical for both structures.",
        "Only the TreeHeap structure tells the decoder which internal subheap should generate the output.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Data Contradiction Rate Without Tree",
        "",
        "```json",
        json.dumps(summary["data"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Boundary",
        "",
        "This proves a controlled S3 generation gate only. It does not prove WMT.",
        "",
    ]
    (evidence_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_same_bag_tree_generation_probe")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vocab-size", type=int, default=64)
    ap.add_argument("--train-triples", type=int, default=2400)
    ap.add_argument("--test-triples", type=int, default=400)
    ap.add_argument("--ood-triples", type=int, default=400)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="treeheap,bow,flat_seq,shape_oracle")
    ap.add_argument("--report-every", type=int, default=10)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(models) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown models: {unknown}; available={sorted(MODEL_REGISTRY)}")

    cfg = Config(
        evidence_dir=args.evidence_dir,
        seed=args.seed,
        vocab_size=args.vocab_size,
        train_triples=args.train_triples,
        test_triples=args.test_triples,
        ood_triples=args.ood_triples,
        dim=args.dim,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        models=models,
        report_every=args.report_every,
    )
    started = time.time()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    arrays = build_split(cfg)
    model_rows = {name: train_model(name, arrays, cfg) for name in models}
    summary = write_outputs(Path(cfg.evidence_dir), cfg, arrays, model_rows, started)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
