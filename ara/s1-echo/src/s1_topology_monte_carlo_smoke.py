"""Small Monte-Carlo topology search smoke for TreeHeap-like aggregation."""

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


def data(n: int, groups: int, leaves: int, dim: int, seed: int, centers=None):
    gen = torch.Generator().manual_seed(seed)
    if centers is None:
        centers = F.normalize(torch.randn(groups, dim, generator=gen), dim=-1)
    y = torch.randint(groups, (n,), generator=gen)
    x = torch.randn(n, leaves, dim, generator=gen) * 0.7
    signal = torch.randint(leaves, (n,), generator=gen)
    for i in range(n):
        x[i, signal[i]] = centers[y[i]] + 0.12 * torch.randn(dim, generator=gen)
    return x, y


TOPOLOGIES = {
    "direct": (0, 1, 2, 3),
    "balanced": ((0, 1), (2, 3)),
    "left_deep": (((0, 1), 2), 3),
    "right_deep": (0, (1, (2, 3))),
    "skip": ((0, 1), 2, 3),
}


def internal_count(node) -> int:
    return 0 if isinstance(node, int) else 1 + sum(internal_count(x) for x in node)


class TopologyModel(nn.Module):
    def __init__(self, topology, dim: int, groups: int):
        super().__init__()
        self.topology = topology
        self.transforms = nn.ModuleList([nn.Linear(dim, dim) for _ in range(internal_count(topology))])
        self.classifier = nn.Linear(dim, groups)

    def fold(self, node, leaves, index):
        if isinstance(node, int):
            return leaves[:, node], index
        children = []
        for child in node:
            value, index = self.fold(child, leaves, index)
            children.append(value)
        base = torch.stack(children, dim=1).mean(dim=1)
        out = base + 0.5 * torch.tanh(self.transforms[index](base))
        return F.normalize(out, dim=-1), index + 1

    def forward(self, leaves):
        parent, _ = self.fold(self.topology, leaves, 0)
        return self.classifier(parent)


def evaluate(name, xtr, ytr, xte, yte, steps, seed, device):
    seed_all(seed)
    model = TopologyModel(TOPOLOGIES[name], xtr.size(-1), int(ytr.max()) + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    for _ in range(steps):
        loss = F.cross_entropy(model(xtr), ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        logits = model(xte)
        acc = float((logits.argmax(-1) == yte).float().mean().item())
    return {"topology": name, "accuracy": acc, "complexity": internal_count(TOPOLOGIES[name]), "loss": float(loss.item()), "finite": bool(torch.isfinite(logits).all().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_gen = torch.Generator().manual_seed(args.seed + 1)
    centers = F.normalize(torch.randn(8, 16, generator=seed_gen), dim=-1)
    xtr, ytr = data(1200, 8, 4, 16, args.seed + 2, centers)
    xte, yte = data(600, 8, 4, 16, args.seed + 3, centers)
    rng = random.Random(args.seed + 3)
    current = "balanced"
    current_score = None
    temperature = 0.15
    rows = []
    for it in range(args.iterations):
        proposal = rng.choice(list(TOPOLOGIES))
        result = evaluate(proposal, xtr, ytr, xte, yte, args.steps, args.seed + 100 + it, device)
        score = result["accuracy"] - 0.01 * result["complexity"]
        accepted = current_score is None or score >= current_score or rng.random() < math.exp((score - current_score) / temperature)
        if accepted:
            current, current_score = proposal, score
        result.update({"iteration": it, "proposal": proposal, "accepted": accepted, "score": score, "current": current, "current_score": current_score, "temperature": temperature})
        rows.append(result)
        temperature *= 0.9
        print(json.dumps(result), flush=True)
    summary = {"seed": args.seed, "device": str(device), "steps_per_candidate": args.steps, "iterations": args.iterations, "initial": "balanced", "final_current": current, "final_score": current_score, "best_observed": max(rows, key=lambda r: r["score"]), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("final_current", "final_score", "best_observed")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
