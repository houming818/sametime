"""Toy embedding probe: co-occurrence background field -> axis transform.

This isolates embedding formation from TreeHeap routing.  It compares an
unmodified background field, an orthogonal axis rotation, and rotation plus
learned per-axis scale.
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


def make_background(vocab: int, groups: int, dim: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(groups, dim, generator=gen), dim=-1)
    labels = torch.arange(vocab) % groups
    x = centers[labels] + 0.30 * torch.randn(vocab, dim, generator=gen)
    # Treat x as a compressed co-occurrence/background field.
    x = F.normalize(x, dim=-1)
    return x, labels


class EmbeddingTransform(nn.Module):
    def __init__(self, mode: str, dim: int):
        super().__init__()
        self.mode = mode
        self.register_buffer("rotation", torch.eye(dim))
        if mode == "rotation_scale":
            self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "background":
            return x
        y = x @ self.rotation.T
        if self.mode == "rotation_scale":
            scale = 0.5 + 1.5 * torch.sigmoid(self.log_scale)
            y = y * scale
        return y


def pair_loss(z: torch.Tensor, labels: torch.Tensor, seed: int):
    gen = torch.Generator(device=z.device).manual_seed(seed)
    n = z.size(0)
    i = torch.randint(n, (n * 8,), generator=gen, device=z.device)
    j = torch.randint(n, (n * 8,), generator=gen, device=gen.device)
    same = labels[i] == labels[j]
    dist = ((z[i] - z[j]).pow(2).sum(-1) + 1e-8).sqrt()
    return (dist[same].pow(2).mean() if same.any() else dist.mean() * 0) + (F.relu(1.5 - dist[~same]).pow(2).mean() if (~same).any() else dist.mean() * 0)


def metrics(z: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        d = torch.cdist(z, z)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        same = same & ~eye
        diff = ~same & ~eye
        same_d = float(d[same].mean().item())
        diff_d = float(d[diff].mean().item())
        nn_idx = d.masked_fill(eye, float("inf")).argmin(-1)
        knn = float((labels[nn_idx] == labels).float().mean().item())
        return {"same_distance": same_d, "different_distance": diff_d, "distance_gap": diff_d - same_d, "knn1_purity": knn, "mean_norm": float(z.norm(dim=-1).mean().item()), "norm_std": float(z.norm(dim=-1).std().item())}


def run(mode: str, x: torch.Tensor, labels: torch.Tensor, steps: int, seed: int, device: torch.device):
    seed_all(seed)
    model = EmbeddingTransform(mode, x.size(-1)).to(device)
    x, labels = x.to(device), labels.to(device)
    if mode in ("background", "rotation"):
        with torch.no_grad():
            z = model(x)
            loss = float(pair_loss(z, labels, seed).item())
        return {"mode": mode, **metrics(z, labels), "final_loss": loss, "final_grad_norm": 0.0, "finite": bool(torch.isfinite(z).all().item()), "trace": []}
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    trace = []
    for step in range(1, steps + 1):
        z = model(x)
        loss = pair_loss(z, labels, seed + step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        opt.step()
        if step in (1, steps) or step % max(1, steps // 4) == 0:
            trace.append({"step": step, "loss": float(loss.item()), "grad_norm": grad})
    z = model(x)
    out = {"mode": mode, **metrics(z, labels), "final_loss": trace[-1]["loss"], "final_grad_norm": trace[-1]["grad_norm"], "finite": bool(torch.isfinite(z).all().item()), "trace": trace}
    if mode == "rotation_scale":
        scale = 0.5 + 1.5 * torch.sigmoid(model.log_scale)
        out["scale_min"] = float(scale.min().item())
        out["scale_max"] = float(scale.max().item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    x, labels = make_background(128, 8, 32, args.seed + 1)
    manifest = {"seed": args.seed, "device": str(device), "vocab": 128, "groups": 8, "dim": 32, "steps": args.steps, "modes": ["background", "rotation", "rotation_scale"], "objective": "same-group compactness and different-group separation", "rotation_constraint": "R^T R = I"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    with (out / "trials.jsonl").open("w", encoding="utf-8") as f:
        for seed_offset in range(args.seeds):
            for mode in manifest["modes"]:
                row = run(mode, x, labels, args.steps, args.seed + 100 + seed_offset, device)
                row["seed"] = args.seed + 100 + seed_offset
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    summary = {}
    fields = ("same_distance", "different_distance", "distance_gap", "knn1_purity", "mean_norm", "norm_std", "final_loss", "final_grad_norm")
    for mode in manifest["modes"]:
        group = [r for r in rows if r["mode"] == mode]
        summary[mode] = {k: sum(float(r[k]) for r in group) / len(group) for k in fields}
    (out / "summary.json").write_text(json.dumps({"manifest": manifest, "summary": summary, "rows": len(rows)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
