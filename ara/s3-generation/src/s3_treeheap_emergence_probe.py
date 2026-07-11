#!/usr/bin/env python3
"""Task-loss-only TreeHeap structural emergence probe.

The model receives no route labels, merge labels, category labels, depth
targets, or compression objective.  It is trained only to generate a two-token
surface answer.  Tree-specific route use is measured afterward through causal
ablations.

For the same ordered leaves [a,b,c]:
  ((a b) c) -> [a,b]
  (a (b c)) -> [b,c]

BoW and flat sequence controls see the same [a,b,c] for both outputs.  The
TreeHeap model sees bracketing only through recursive composition, never as a
shape-bit feature.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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
    report_every: int
    models: List[str]


def make_ordered_triples(vocab_size: int, seed: int) -> List[Tuple[int, int, int]]:
    """Split by unordered token set so a mirrored OOD triple is never in train."""
    rng = random.Random(seed)
    triples: List[Tuple[int, int, int]] = []
    for combo in itertools.combinations(range(1, vocab_size), 3):
        ordered = list(combo)
        rng.shuffle(ordered)
        triples.append(tuple(ordered))
    rng.shuffle(triples)
    return triples


def targets_for(x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    left_pair = x[:, :2]
    right_pair = x[:, 1:]
    return torch.where(shape[:, None] == 0, left_pair, right_pair)


def build_split(cfg: Config) -> Dict[str, torch.Tensor]:
    need = cfg.train_triples + cfg.test_triples + cfg.ood_triples
    triples = make_ordered_triples(cfg.vocab_size, cfg.seed)
    if len(triples) < need:
        raise ValueError(f"need {need} unordered triples, only have {len(triples)}")

    selected = {
        "train": triples[: cfg.train_triples],
        "test": triples[cfg.train_triples : cfg.train_triples + cfg.test_triples],
        "ood": triples[cfg.train_triples + cfg.test_triples : need],
    }
    out: Dict[str, torch.Tensor] = {}
    for split, rows in selected.items():
        leaves = torch.tensor([item for triple in rows for item in (triple, triple)], dtype=torch.long)
        shapes = torch.tensor([shape for _ in rows for shape in (0, 1)], dtype=torch.long)
        out[f"{split}_x"] = leaves
        out[f"{split}_shape"] = shapes
        out[f"{split}_y"] = targets_for(leaves, shapes)
    return out


class PairDecoder(nn.Module):
    def __init__(self, dim: int, hidden: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).view(state.shape[0], 2, self.vocab_size)


class BowGenerator(nn.Module):
    kind = "bow"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.decoder = PairDecoder(dim, hidden, vocab_size)

    def forward(self, x: torch.Tensor, shape: torch.Tensor, intervention: str = "full") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del shape, intervention
        return self.decoder(self.emb(x).mean(dim=1)), {}


class FlatSequenceGenerator(nn.Module):
    kind = "flat_seq"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.project = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.decoder = PairDecoder(dim, hidden, vocab_size)

    def forward(self, x: torch.Tensor, shape: torch.Tensor, intervention: str = "full") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del shape, intervention
        state = self.project((self.emb(x) + self.pos[None]).reshape(x.shape[0], -1))
        return self.decoder(state), {}


class ShapeOracleGenerator(nn.Module):
    """Visible-structure control, not a TreeHeap competitor."""

    kind = "shape_oracle"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.shape = nn.Embedding(2, dim)
        self.project = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.decoder = PairDecoder(dim, hidden, vocab_size)

    def forward(self, x: torch.Tensor, shape: torch.Tensor, intervention: str = "full") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del intervention
        state = torch.cat([(self.emb(x) + self.pos[None]).reshape(x.shape[0], -1), self.shape(shape)], dim=-1)
        return self.decoder(self.project(state)), {}


class TreeHeapGenerator(nn.Module):
    kind = "treeheap"

    def __init__(self, vocab_size: int, dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.left_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.right_role = nn.Parameter(torch.randn(dim) * 0.02)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
            nn.Tanh(),
        )
        self.route = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 3),  # stop, left, right
        )
        self.decoder = PairDecoder(dim, hidden, vocab_size)

    def plus(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # This is the local non-commutative TreeHeap convolution.
        return self.compose(torch.cat([left + self.left_role, right + self.right_role], dim=-1))

    def states(self, x: torch.Tensor, shape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, c = self.emb(x[:, 0]), self.emb(x[:, 1]), self.emb(x[:, 2])
        ab = self.plus(a, b)
        bc = self.plus(b, c)
        left = torch.where(shape[:, None] == 0, ab, a)
        right = torch.where(shape[:, None] == 0, c, bc)
        return self.plus(left, right), left, right

    def forward(self, x: torch.Tensor, shape: torch.Tensor, intervention: str = "full") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        root, left, right = self.states(x, shape)
        route_input = torch.cat([root, left, right, self.query.expand(x.shape[0], -1)], dim=-1)
        probabilities = F.softmax(self.route(route_input), dim=-1)

        if intervention == "root_only":
            read = root
        else:
            read_left, read_right = left, right
            if intervention == "zero_internal":
                read_left = torch.where(shape[:, None] == 0, torch.zeros_like(left), left)
                read_right = torch.where(shape[:, None] == 1, torch.zeros_like(right), right)
            read = (
                probabilities[:, 0:1] * root
                + probabilities[:, 1:2] * read_left
                + probabilities[:, 2:3] * read_right
            )
        return self.decoder(read), {"route_probs": probabilities}


MODEL_REGISTRY = {
    "bow": BowGenerator,
    "flat_seq": FlatSequenceGenerator,
    "shape_oracle": ShapeOracleGenerator,
    "treeheap": TreeHeapGenerator,
}


def batch_indices(n: int, batch_size: int, rng: random.Random) -> Iterable[List[int]]:
    indices = list(range(n))
    rng.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def basic_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    return {
        "loss": float(loss_fn(logits, target).item()),
        "token_acc": float((pred == target).float().mean().item()),
        "exact": float((pred == target).all(dim=1).float().mean().item()),
        "first_token_acc": float((pred[:, 0] == target[:, 0]).float().mean().item()),
        "second_token_acc": float((pred[:, 1] == target[:, 1]).float().mean().item()),
    }


def route_metrics(aux: Dict[str, torch.Tensor], shape: torch.Tensor) -> Dict[str, float]:
    if "route_probs" not in aux:
        return {}
    probs = aux["route_probs"]
    expected_internal = torch.where(shape == 0, torch.ones_like(shape), torch.full_like(shape, 2))
    return {
        "route_internal_acc": float((probs.argmax(dim=-1) == expected_internal).float().mean().item()),
        "route_internal_probability": float(probs.gather(1, expected_internal[:, None]).mean().item()),
        "route_stop_probability": float(probs[:, 0].mean().item()),
        "route_entropy_bits": float((-(probs * probs.clamp_min(1e-9).log2()).sum(dim=-1)).mean().item()),
    }


def evaluate(model: nn.Module, x: torch.Tensor, shape: torch.Tensor, target: torch.Tensor, intervention: str = "full") -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, aux = model(x, shape, intervention)
        result = basic_metrics(logits, target)
        result.update(route_metrics(aux, shape))
    return result


def mirror_inputs(x: torch.Tensor, shape: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mirrored_x = x.flip(dims=[1])
    mirrored_shape = 1 - shape
    return mirrored_x, mirrored_shape, targets_for(mirrored_x, mirrored_shape)


def train_one(name: str, arrays: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, object]:
    seed_offset = sum(ord(ch) for ch in name)
    torch.manual_seed(cfg.seed + seed_offset)
    rng = random.Random(cfg.seed + 1000 + seed_offset)
    model = MODEL_REGISTRY[name](cfg.vocab_size, cfg.dim, cfg.hidden).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_x = arrays["train_x"].to(cfg.device)
    train_shape = arrays["train_shape"].to(cfg.device)
    train_y = arrays["train_y"].to(cfg.device)
    test_x = arrays["test_x"].to(cfg.device)
    test_shape = arrays["test_shape"].to(cfg.device)
    test_y = arrays["test_y"].to(cfg.device)
    ood_x = arrays["ood_x"].to(cfg.device)
    ood_shape = arrays["ood_shape"].to(cfg.device)
    ood_y = arrays["ood_y"].to(cfg.device)
    trace: List[Dict[str, float | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for indices in batch_indices(train_x.shape[0], cfg.batch_size, rng):
            index = torch.tensor(indices, dtype=torch.long, device=cfg.device)
            logits, _ = model(train_x[index], train_shape[index])
            loss = loss_fn(logits, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(indices)
            seen += len(indices)

        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.report_every == 0:
            row: Dict[str, float | int] = {"epoch": epoch, "train_loss": weighted_loss / max(1, seen)}
            for prefix, x, shape, y in (("test", test_x, test_shape, test_y), ("ood", ood_x, ood_shape, ood_y)):
                for metric, value in evaluate(model, x, shape, y).items():
                    row[f"{prefix}_{metric}"] = value
            trace.append(row)
            print(f"[{name}] epoch={epoch} loss={row['train_loss']:.4f} ood_exact={row['ood_exact']:.4f}", flush=True)

    metrics = {
        "train": evaluate(model, train_x, train_shape, train_y),
        "test": evaluate(model, test_x, test_shape, test_y),
        "ood": evaluate(model, ood_x, ood_shape, ood_y),
    }
    if name == "treeheap":
        metrics["ood_root_only"] = evaluate(model, ood_x, ood_shape, ood_y, "root_only")
        metrics["ood_zero_internal"] = evaluate(model, ood_x, ood_shape, ood_y, "zero_internal")
        mx, ms, my = mirror_inputs(ood_x, ood_shape)
        metrics["ood_mirror"] = evaluate(model, mx, ms, my)

    examples = []
    model.eval()
    with torch.no_grad():
        logits, aux = model(ood_x[:8], ood_shape[:8])
        prediction = logits.argmax(dim=-1).cpu()
        probs = aux.get("route_probs")
        for index in range(min(8, ood_x.shape[0])):
            examples.append(
                {
                    "leaves": ood_x[index].cpu().tolist(),
                    "shape": int(ood_shape[index].cpu().item()),
                    "target": ood_y[index].cpu().tolist(),
                    "prediction": prediction[index].tolist(),
                    "route_probs": [] if probs is None else [round(float(x), 6) for x in probs[index].cpu().tolist()],
                }
            )
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trace": trace,
        "metrics": metrics,
        "examples": examples,
    }


def first_emergence_epoch(trace: Sequence[Dict[str, float | int]]) -> int | None:
    for row in trace:
        if row.get("ood_exact", 0.0) >= 0.90 and row.get("ood_route_internal_acc", 0.0) >= 0.90:
            return int(row["epoch"])
    return None


def compact_summary(rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for name, row in rows.items():
        result[name] = {"parameters": row["parameters"], "ood": row["metrics"]["ood"]}  # type: ignore[index]
    if "treeheap" in rows:
        tree_metrics = rows["treeheap"]["metrics"]  # type: ignore[index]
        result["treeheap_causal"] = {
            "full_ood_exact": tree_metrics["ood"]["exact"],
            "root_only_ood_exact": tree_metrics["ood_root_only"]["exact"],
            "zero_internal_ood_exact": tree_metrics["ood_zero_internal"]["exact"],
            "mirror_ood_exact": tree_metrics["ood_mirror"]["exact"],
            "root_only_drop": tree_metrics["ood"]["exact"] - tree_metrics["ood_root_only"]["exact"],
            "zero_internal_drop": tree_metrics["ood"]["exact"] - tree_metrics["ood_zero_internal"]["exact"],
            "first_emergence_epoch": first_emergence_epoch(rows["treeheap"]["trace"]),  # type: ignore[index]
        }
    if all(name in rows for name in ("treeheap", "bow", "flat_seq")):
        tree = rows["treeheap"]["metrics"]["ood"]  # type: ignore[index]
        bow = rows["bow"]["metrics"]["ood"]  # type: ignore[index]
        flat = rows["flat_seq"]["metrics"]["ood"]  # type: ignore[index]
        result["treeheap_minus_structure_blind"] = {
            "exact_minus_bow": tree["exact"] - bow["exact"],
            "exact_minus_flat_seq": tree["exact"] - flat["exact"],
            "token_acc_minus_bow": tree["token_acc"] - bow["token_acc"],
            "token_acc_minus_flat_seq": tree["token_acc"] - flat["token_acc"],
        }
    return result


def write_evidence(evidence_dir: Path, cfg: Config, arrays: Dict[str, torch.Tensor], rows: Dict[str, Dict[str, object]], started: float) -> None:
    summary = {
        "claim": "S3-TREEHEAP-EMERGENCE-C01",
        "experiment": "P-S3-TREEHEAP-EMERGENCE01",
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": asdict(cfg),
        "data": {
            "train_items": int(arrays["train_x"].shape[0]),
            "test_items": int(arrays["test_x"].shape[0]),
            "ood_items": int(arrays["ood_x"].shape[0]),
            "same_ordered_leaves_have_two_targets": True,
            "route_labels_used_in_training": False,
            "depth_or_compression_loss_used": False,
        },
        "models": rows,
        "summary": compact_summary(rows),
        "decision_hint": {
            "support_if": "TreeHeap OOD exact beats both structure-blind controls, route audit selects internal child, and root-only/zero-internal interventions materially reduce OOD exact.",
            "not_proved": "natural-language parsing, semantic Huffman compression, a universal loss threshold, WMT translation, or superiority over arbitrary models given equivalent tree structure.",
        },
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (evidence_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for name, row in rows.items():
            for trace_row in row["trace"]:  # type: ignore[index]
                handle.write(json.dumps({"model": name, **trace_row}, ensure_ascii=False) + "\n")
    (evidence_dir / "README.md").write_text(
        "# Task-Loss TreeHeap Structural Emergence Probe\n\n"
        "Only surface cross-entropy is optimized. Route and structural metrics are audits.\n\n"
        "```json\n" + json.dumps(summary["summary"], indent=2, ensure_ascii=False) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_treeheap_emergence_probe")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--train-triples", type=int, default=6000)
    parser.add_argument("--test-triples", type=int, default=1000)
    parser.add_argument("--ood-triples", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--models", default="treeheap,bow,flat_seq,shape_oracle")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"unknown models: {unknown}; available={sorted(MODEL_REGISTRY)}")
    cfg = Config(**{**vars(args), "models": models})
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    arrays = build_split(cfg)
    started = time.time()
    rows = {name: train_one(name, arrays, cfg) for name in cfg.models}
    write_evidence(Path(cfg.evidence_dir), cfg, arrays, rows, started)


if __name__ == "__main__":
    main()
