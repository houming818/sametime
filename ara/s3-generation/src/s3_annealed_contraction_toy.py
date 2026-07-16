#!/usr/bin/env python3
"""Controlled proof of predictive-core survival under TreeHeap contraction."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


CORE_VALUES = 8
NUISANCE_VALUES = 32
INPUT_LENGTH = 16
CORE_COUNT = 3


def make_split(count: int, seed: int, split: str) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    rows: List[List[int]] = []
    targets: List[List[int]] = []
    positions: List[List[int]] = []
    nuisance_positions: List[List[int]] = []
    while len(rows) < count:
        a, b, c = (int(x) for x in rng.integers(0, CORE_VALUES, size=3))
        heldout = ((a * 64 + b * 8 + c) % 5) == 0
        if (split == "test") != heldout:
            continue
        core_pos = sorted(int(x) for x in rng.choice(INPUT_LENGTH, CORE_COUNT, replace=False))
        row = [48 + int(x) for x in rng.integers(0, NUISANCE_VALUES, size=INPUT_LENGTH)]
        for pos, token in zip(core_pos, (a, 8 + b, 16 + c)):
            row[pos] = token
        noise_pos = [index for index in range(INPUT_LENGTH) if index not in core_pos]
        rows.append(row)
        positions.append(core_pos)
        nuisance_positions.append(noise_pos[:CORE_COUNT])
        targets.append([(a + b) % 8, (b + c) % 8, (a + c) % 8])
    return {
        "tokens": torch.tensor(rows, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.long),
        "core_positions": torch.tensor(positions, dtype=torch.long),
        "nuisance_positions": torch.tensor(nuisance_positions, dtype=torch.long),
    }


class FutureHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3 * CORE_VALUES))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        score = (nodes * self.query).sum(-1) / math.sqrt(nodes.shape[-1])
        context = (score.softmax(-1)[..., None] * nodes).sum(1)
        return self.out(context).reshape(-1, 3, CORE_VALUES)


class TreeModel(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.fold = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.depth = nn.Embedding(5, dim)
        self.head = FutureHead(dim)

    def frontiers(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        node = self.embedding(tokens)
        levels = [node]
        depth = 0
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            update = self.fold(torch.cat((left, right), dim=-1))
            node = self.norm((left + right) * math.sqrt(0.5) + update + self.depth.weight[depth])
            levels.append(node)
            depth += 1
        return levels

    def forward(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        return self.head(self.frontiers(tokens)[depth])

    def root(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.frontiers(tokens)[-1][:, 0]


class MeanModel(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.head = FutureHead(dim)

    def forward(self, tokens: torch.Tensor, depth: int = -1) -> torch.Tensor:
        return self.head(self.embedding(tokens).mean(1, keepdim=True))

    def root(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens).mean(1)


class FlatModel(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.bottleneck = nn.Sequential(nn.Linear(INPUT_LENGTH * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = FutureHead(dim)

    def root(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.embedding(tokens).flatten(1))

    def forward(self, tokens: torch.Tensor, depth: int = -1) -> torch.Tensor:
        return self.head(self.root(tokens)[:, None])


def batches(data: Dict[str, torch.Tensor], batch: int, generator: torch.Generator):
    order = torch.randperm(data["tokens"].shape[0], generator=generator)
    for start in range(0, len(order), batch):
        index = order[start : start + batch]
        yield {key: value[index] for key, value in data.items()}


def annealed_depth(step: int, total: int, depths: int, rng: random.Random) -> int:
    progress = step / max(1, total - 1)
    direction = 1.0 - 2.0 * progress
    temperature = 1.5 * (0.25 / 1.5) ** progress
    logits = [direction * depth / temperature for depth in range(depths)]
    maximum = max(logits)
    soft = [math.exp(value - maximum) for value in logits]
    soft = [value / sum(soft) for value in soft]
    probability = [0.25 / depths + 0.75 * value for value in soft]
    return rng.choices(range(depths), weights=probability)[0]


@torch.no_grad()
def metrics(model, data: Dict[str, torch.Tensor], device: str, depth: int = -1, batch: int = 512) -> dict:
    model.eval()
    total_loss = total = exact = token = 0
    for start in range(0, data["tokens"].shape[0], batch):
        x = data["tokens"][start : start + batch].to(device)
        y = data["target"][start : start + batch].to(device)
        logits = model(x, depth)
        total_loss += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
        total += y.numel()
        prediction = logits.argmax(-1)
        token += int(prediction.eq(y).sum())
        exact += int(prediction.eq(y).all(-1).sum())
    return {"nll": total_loss / total, "token_accuracy": token / total, "exact": exact / data["tokens"].shape[0]}


@torch.no_grad()
def intervention(model, data: Dict[str, torch.Tensor], device: str, kind: str) -> dict:
    model.eval()
    x = data["tokens"].clone()
    positions = data[f"{kind}_positions"]
    donor_tokens = data["tokens"].roll(1, dims=0)
    donor_positions = positions.roll(1, dims=0)
    for row in range(x.shape[0]):
        values = donor_tokens[row].gather(0, donor_positions[row])
        x[row].scatter_(0, positions[row], values)
    changed = dict(data)
    changed["tokens"] = x
    return metrics(model, changed, device, -1)


def probe(root_train: torch.Tensor, train: Dict[str, torch.Tensor], root_test: torch.Tensor, test: Dict[str, torch.Tensor], device: str) -> dict:
    core_labels = torch.stack((
        train["tokens"].gather(1, train["core_positions"])[:, 0],
        train["tokens"].gather(1, train["core_positions"])[:, 1] - 8,
        train["tokens"].gather(1, train["core_positions"])[:, 2] - 16,
    ), dim=1)
    test_core = torch.stack((
        test["tokens"].gather(1, test["core_positions"])[:, 0],
        test["tokens"].gather(1, test["core_positions"])[:, 1] - 8,
        test["tokens"].gather(1, test["core_positions"])[:, 2] - 16,
    ), dim=1)
    nuisance = train["tokens"].gather(1, train["nuisance_positions"])[:, 0] - 48
    test_nuisance = test["tokens"].gather(1, test["nuisance_positions"])[:, 0] - 48
    core_head = nn.Linear(root_train.shape[1], 3 * CORE_VALUES).to(device)
    nuisance_head = nn.Linear(root_train.shape[1], NUISANCE_VALUES).to(device)
    opt = torch.optim.AdamW(list(core_head.parameters()) + list(nuisance_head.parameters()), lr=0.02)
    root_train, core_labels, nuisance = root_train.to(device), core_labels.to(device), nuisance.to(device)
    generator = torch.Generator().manual_seed(991)
    for _ in range(300):
        index = torch.randint(0, root_train.shape[0], (512,), generator=generator).to(device)
        core_logits = core_head(root_train[index]).reshape(-1, 3, CORE_VALUES)
        loss = F.cross_entropy(core_logits.flatten(0, 1), core_labels[index].flatten())
        loss = loss + F.cross_entropy(nuisance_head(root_train[index]), nuisance[index])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        core_pred = core_head(root_test.to(device)).reshape(-1, 3, CORE_VALUES).argmax(-1).cpu()
        nuisance_pred = nuisance_head(root_test.to(device)).argmax(-1).cpu()
    return {
        "core_accuracy": float(core_pred.eq(test_core).float().mean()),
        "nuisance_accuracy": float(nuisance_pred.eq(test_nuisance).float().mean()),
    }


def train_one(name: str, seed: int, args, train: Dict[str, torch.Tensor], test: Dict[str, torch.Tensor]) -> Tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    cls = {"tree": TreeModel, "mean": MeanModel, "flat": FlatModel}[name]
    model = cls(48 + NUISANCE_VALUES, args.dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(seed + 71)
    generator = torch.Generator().manual_seed(seed + 91)
    iterator = iter(batches(train, args.batch, generator))
    trace = []
    for step in range(args.steps):
        try: row = next(iterator)
        except StopIteration:
            iterator = iter(batches(train, args.batch, generator)); row = next(iterator)
        x, y = row["tokens"].to(args.device), row["target"].to(args.device)
        depth = annealed_depth(step, args.steps, 5, rng) if name == "tree" else -1
        logits = model(x, depth)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            root_metric = metrics(model, test, args.device, -1)
            trace.append({"step": step + 1, "train_nll": float(loss.detach()), "root": root_metric})
            print(json.dumps({"seed": seed, "model": name, **trace[-1]}), flush=True)
    result = {"parameters": sum(p.numel() for p in model.parameters()), "trace": trace, "root": metrics(model, test, args.device, -1)}
    if name == "tree":
        result["frontiers"] = [metrics(model, test, args.device, depth) for depth in range(5)]
        native = result["root"]["nll"]
        core = intervention(model, test, args.device, "core")
        nuisance = intervention(model, test, args.device, "nuisance")
        result["interventions"] = {
            "core_shuffle": {**core, "damage": core["nll"] - native},
            "nuisance_shuffle": {**nuisance, "damage": nuisance["nll"] - native},
        }
        with torch.no_grad():
            root_train = model.root(train["tokens"][: min(12000, len(train["tokens"]))].to(args.device)).cpu()
            root_test = model.root(test["tokens"].to(args.device)).cpu()
        result["probe"] = probe(root_train, {k: v[:len(root_train)] for k, v in train.items()}, root_test, test, args.device)
    return model, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_annealed_contraction_toy")
    parser.add_argument("--seeds", nargs="+", type=int, default=[72001, 72002, 72003])
    parser.add_argument("--train-samples", type=int, default=24000)
    parser.add_argument("--test-samples", type=int, default=4000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    started = time.time(); results = {}
    for seed in args.seeds:
        train = make_split(args.train_samples, seed, "train")
        test = make_split(args.test_samples, seed + 1000, "test")
        seed_result = {}
        models = {}
        for name in ("tree", "mean", "flat"):
            models[name], seed_result[name] = train_one(name, seed, args, train, test)
        tree = seed_result["tree"]
        core_margin = tree["interventions"]["core_shuffle"]["damage"] - tree["interventions"]["nuisance_shuffle"]["damage"]
        seed_result["derived"] = {
            "core_over_nuisance_damage": core_margin,
            "core_probe_margin": tree["probe"]["core_accuracy"] - tree["probe"]["nuisance_accuracy"],
            "tree_exact_gain_over_best_control": tree["root"]["exact"] - max(seed_result["mean"]["root"]["exact"], seed_result["flat"]["root"]["exact"]),
        }
        seed_result["gates"] = {
            "T1_root_exact": tree["root"]["exact"] >= 0.90,
            "T2_core_causal": core_margin >= 0.50,
            "T3_core_probe": seed_result["derived"]["core_probe_margin"] >= 0.30,
            "T4_beats_controls": seed_result["derived"]["tree_exact_gain_over_best_control"] >= 0.05,
        }
        results[str(seed)] = seed_result
        torch.save({"state_dict": models["tree"].state_dict(), "config": vars(args), "seed": seed}, output / f"checkpoint_tree_{seed}.pt")
        del models
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
    two_seed = sum(all(row["gates"][key] for key in ("T1_root_exact", "T2_core_causal", "T3_core_probe")) for row in results.values()) >= 2
    summary = {
        "claim": "S3-ANNEAL-TOY-C01",
        "host": socket.gethostname(), "seconds": time.time() - started,
        "config": vars(args), "results": results,
        "cross_seed_gate_T5": two_seed,
        "decision": "supported" if two_seed else "not supported or partial",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text("# Annealed Contraction Toy\n\nComplete-input predictive-core proof; no core labels enter training loss.\n", encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "T5": two_seed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
