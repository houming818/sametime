"""Toy probe for dynamic child integration in a TreeHeap-like parent.

This is an aggregation experiment, not a language-model result.  Each sample
contains several child vectors; one child carries the group signal and the
others are distractors.  We compare mean pooling with learned routing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(n: int, groups: int, children: int, dim: int, seed: int, centers: torch.Tensor | None = None):
    gen = torch.Generator().manual_seed(seed)
    if centers is None:
        centers = F.normalize(torch.randn(groups, dim, generator=gen), dim=-1)
    labels = torch.randint(groups, (n,), generator=gen)
    signal_pos = torch.randint(children, (n,), generator=gen)
    x = torch.randn(n, children, dim, generator=gen) * 0.7
    for i in range(n):
        g = int(labels[i])
        p = int(signal_pos[i])
        x[i, p] = centers[g] + 0.12 * torch.randn(dim, generator=gen)
    return x, labels, signal_pos, centers


class ParentF(nn.Module):
    def __init__(self, mode: str, dim: int, groups: int, children: int, alpha: float):
        super().__init__()
        self.mode = mode
        self.num_children = children
        self.alpha = alpha
        self.classifier = nn.Linear(dim, groups)
        if mode != "mean":
            if mode.startswith("orthogonal"):
                self.rotation = nn.Linear(dim, dim, bias=False)
                nn.init.eye_(self.rotation.weight)
                nn.utils.parametrizations.orthogonal(self.rotation)
            else:
                self.query = nn.Linear(dim, dim, bias=False)
                self.key = nn.Linear(dim, dim, bias=False)
            self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        base = x.mean(dim=1)
        if self.mode == "mean":
            parent = base
            weights = torch.full((x.size(0), x.size(1)), 1.0 / x.size(1), device=x.device)
        else:
            if self.mode.startswith("orthogonal"):
                q = self.rotation(base).unsqueeze(1)
                k = self.rotation(x)
            else:
                q = self.query(base).unsqueeze(1)
                k = self.key(x)
            scores = (q * k).sum(-1) / math.sqrt(x.size(-1))
            weights = F.softmax(scores / self.temperature.clamp_min(0.05), dim=-1)
            routed = (weights.unsqueeze(-1) * x).sum(dim=1)
            if self.mode == "dynamic":
                parent = routed
            else:
                parent = base + self.alpha * (routed - base)
        parent = F.normalize(parent, dim=-1)
        return self.classifier(parent), weights, parent


def entropy(weights: torch.Tensor) -> float:
    h = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
    return float((h / math.log(weights.size(-1))).mean().item())


def run(mode: str, xtr, ytr, ptr, xte, yte, pte, groups, dim, children, steps, seed, device, alpha):
    seed_all(seed)
    model = ParentF(mode, dim, groups, children, alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    xtr, ytr = xtr.to(device), ytr.to(device)
    xte, yte = xte.to(device), yte.to(device)
    trace = []
    for step in range(1, steps + 1):
        logits, weights, _ = model(xtr)
        loss = F.cross_entropy(logits, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        opt.step()
        if step in (1, steps) or step % max(1, steps // 4) == 0:
            trace.append({"step": step, "loss": float(loss.item()), "grad_norm": grad})
    with torch.no_grad():
        logits, weights, _ = model(xte)
        acc = float((logits.argmax(-1) == yte).float().mean().item())
        top1 = float((weights.argmax(-1) == pte.to(device)).float().mean().item()) if mode != "mean" else 1.0 / children
        result = {
            "mode": mode,
            "accuracy": acc,
            "route_entropy": entropy(weights),
            "signal_top1": top1,
            "final_loss": trace[-1]["loss"],
            "final_grad_norm": trace[-1]["grad_norm"],
            "finite": bool(torch.isfinite(logits).all().item()),
            "trace": trace,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    _, _, _, centers = make_data(1, 8, 8, 32, args.seed + 1)
    xtr, ytr, ptr, _ = make_data(2400, 8, 8, 32, args.seed + 2, centers)
    xte, yte, pte, _ = make_data(1200, 8, 8, 32, args.seed + 3, centers)
    manifest = {"seed": args.seed, "device": str(device), "groups": 8, "children": 8, "dim": 32, "train": 2400, "test": 1200, "steps": args.steps, "modes": ["mean", "dynamic", "dynamic_residual", "orthogonal", "orthogonal_residual"], "alpha": 0.5, "orthogonal_constraint": "R^T R = I"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (out / "trials.jsonl").open("w", encoding="utf-8") as f:
        for seed_offset in range(args.seeds):
            for mode in manifest["modes"]:
                row = run(mode, xtr, ytr, ptr, xte, yte, pte, 8, 32, 8, args.steps, args.seed + 100 + seed_offset, device, 0.5)
                row["seed"] = args.seed + 100 + seed_offset
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    summary = {}
    for mode in manifest["modes"]:
        group = [r for r in rows if r["mode"] == mode]
        summary[mode] = {k: sum(float(r[k]) for r in group) / len(group) for k in ("accuracy", "route_entropy", "signal_top1", "final_loss", "final_grad_norm")}
    (out / "summary.json").write_text(json.dumps({"manifest": manifest, "summary": summary, "rows": len(rows)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
