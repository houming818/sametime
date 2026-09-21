"""Overnight exploratory search over candidate simplex update operators F.

This is a toy search, not a language-model result.  It keeps one synthetic
relation task fixed and records every candidate independently.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as torch_f


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pairs(vocab: int, groups: int, train_n: int, test_n: int, seed: int):
    rng = random.Random(seed)

    def sample(n: int):
        rows = []
        for _ in range(n):
            group = rng.randrange(groups)
            center = group * (vocab // groups) + rng.randrange(vocab // groups)
            positive = group * (vocab // groups) + rng.randrange(vocab // groups)
            negative_group = (group + rng.randrange(1, groups)) % groups
            negative = negative_group * (vocab // groups) + rng.randrange(vocab // groups)
            rows.append((center, positive, negative))
        return torch.tensor(rows, dtype=torch.long)

    return sample(train_n), sample(test_n)


class Candidate(nn.Module):
    def __init__(self, family: str, leaves: int, width: int, residual: float):
        super().__init__()
        self.family = family
        self.leaves = leaves
        self.residual = residual
        if family == "linear":
            self.body = nn.Linear(leaves, leaves)
        elif family == "poly2":
            self.body = nn.Linear(leaves * 2, leaves)
        elif family == "mlp":
            self.body = nn.Sequential(nn.Linear(leaves, width), nn.Tanh(), nn.Linear(width, leaves))
        elif family == "energy":
            self.body = nn.Linear(leaves, leaves)
        else:
            raise ValueError(f"unknown family: {family}")
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def apply_once(self, state: torch.Tensor) -> torch.Tensor:
        if self.family == "poly2":
            logits = self.body(torch.cat((state, state.square()), dim=-1))
        else:
            logits = self.body(state)
        if self.family == "energy":
            logits = state.clamp_min(1e-7).log() + logits
        proposal = torch_f.softmax(logits / self.temperature.clamp_min(0.05), dim=-1)
        mixed = (1.0 - self.residual) * state + self.residual * proposal
        return mixed.clamp_min(1e-8) / mixed.sum(dim=-1, keepdim=True)


def js_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (a + b)
    return 0.5 * (a * (a / m).clamp_min(1e-8).log()).sum(-1) + 0.5 * (
        b * (b / m).clamp_min(1e-8).log()
    ).sum(-1)


def evaluate(model, base, pairs, rounds):
    with torch.no_grad():
        state = base
        for _ in range(rounds):
            state = model.apply_once(state)
        c, p, n = pairs.T
        pos = (state[c] * state[p]).sum(-1)
        neg = (state[c] * state[n]).sum(-1)
        accuracy = (pos > neg).float().mean().item()
        leaf = state.argmax(-1)
        used = torch.bincount(leaf, minlength=state.shape[-1]).float()
        used = used[used > 0]
        prob = used / used.sum()
        entropy = float((-(prob * prob.clamp_min(1e-8).log2()).sum() / math.log2(state.shape[-1])).item())
        return {
            "accuracy": accuracy,
            "margin": float((pos - neg).mean().item()),
            "leaf_utilization": float(len(used) / state.shape[-1]),
            "occupancy_entropy": entropy,
            "mean_max_probability": float(state.max(-1).values.mean().item()),
            "state_js_from_base": float(js_distance(state, base).mean().item()),
            "finite": bool(torch.isfinite(state).all().item()),
        }


def run_trial(spec, train_pairs, test_pairs, device, steps, batch_size, seed):
    set_seed(seed)
    vocab = int(max(train_pairs.max(), test_pairs.max()).item()) + 1
    leaves = spec["leaves"]
    base_logits = torch.randn(vocab, leaves, device=device) * 0.15
    base = torch_f.softmax(base_logits, dim=-1).detach()
    model = Candidate(spec["family"], leaves, spec["width"], spec["residual"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"])
    train_pairs = train_pairs.to(device)
    test_pairs = test_pairs.to(device)
    rng = torch.Generator(device="cpu").manual_seed(seed + 17)
    trace = []
    started = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(len(train_pairs), (batch_size,), generator=rng)
        c, p, n = train_pairs[idx].T
        state = base
        for _ in range(spec["rounds"]):
            state = model.apply_once(state)
        pos = (state[c] * state[p]).sum(-1)
        neg = (state[c] * state[n]).sum(-1)
        loss = -torch_f.logsigmoid(8.0 * (pos - neg)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        opt.step()
        if step in (1, steps) or step % max(1, steps // 10) == 0:
            trace.append({"step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm})
        if not math.isfinite(float(loss.detach().item())):
            raise FloatingPointError("non-finite loss")
    metrics = evaluate(model, base, test_pairs, spec["rounds"])
    metrics["elapsed_seconds"] = time.time() - started
    metrics["final_loss"] = trace[-1]["loss"]
    return metrics, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--trials", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_seed(args.seed)
    train_pairs, test_pairs = make_pairs(64, 8, 1200, 800, args.seed + 1)
    families = ["linear", "poly2", "mlp", "energy"]
    matrix = list(itertools.product(families, (8, 16), (16, 32), (0.25, 0.5, 0.75), (0.5, 1.0)))
    rng = random.Random(args.seed + 2)
    rng.shuffle(matrix)
    matrix = matrix[: args.trials]
    manifest = {
        "seed": args.seed,
        "device": str(device),
        "toy_vocab": 64,
        "toy_groups": 8,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "matrix_size": len(matrix),
        "axes": {"family": families, "leaves": [8, 16], "width": [16, 32], "residual": [0.25, 0.5, 0.75], "lr": [0.5, 1.0]},
        "steps_per_trial": args.steps,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    with (out / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for index, (family, leaves, width, residual, lr_scale) in enumerate(matrix):
            spec = {"family": family, "leaves": leaves, "width": width, "residual": residual, "lr": 0.002 * lr_scale, "rounds": 4 if index % 3 == 0 else (2 if index % 3 == 1 else 1)}
            row = {"trial": index, **spec}
            try:
                metrics, trace = run_trial(spec, train_pairs, test_pairs, device, args.steps, 128, args.seed + 1000 + index)
                row.update(metrics)
                row["trace"] = trace
                row["status"] = "done"
            except Exception as exc:
                row.update({"status": "failed", "error": repr(exc)})
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            results.append(row)
            print(json.dumps(row), flush=True)
    valid = [r for r in results if r.get("status") == "done" and r.get("finite")]
    valid.sort(key=lambda r: (r["accuracy"], r["occupancy_entropy"]), reverse=True)
    summary = {"manifest": manifest, "completed": len(valid), "failed": len(results) - len(valid), "top5": valid[:5]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
