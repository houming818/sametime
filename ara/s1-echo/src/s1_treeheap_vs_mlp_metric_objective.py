#!/usr/bin/env python3
"""Compare TreeHeap and MLP under one goal, with architecture-native codes."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import s1_encoder_minimal_observer_probe as observer
import s1_treeheap_metric_self_improvement as tree_metric


class MetricMLP(nn.Module):
    def __init__(self, n_verb: int, n_obj: int, dim: int, hidden: int):
        super().__init__()
        self.n_obj = n_obj
        self.dim = dim
        self.object_state = nn.Parameter(torch.randn(n_obj, dim) * 0.08)
        self.verb_state = nn.Parameter(torch.randn(n_verb, dim) * 0.08)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.echo_head = nn.Linear(dim, n_obj)
        self.object_bias = nn.Parameter(torch.zeros(n_obj))

    def codes(self):
        return F.normalize(self.encoder(self.object_state), dim=-1, eps=1e-8)

    def context_logits(self):
        return self.verb_state @ self.codes().t() / math.sqrt(self.dim) + self.object_bias[None]

    def echo_logits(self):
        return self.echo_head(self.object_state)


def codes_and_distance(model, architecture):
    if architecture == "treeheap":
        codes = tree_metric.treeheap_codes(model)
        distance = tree_metric.pairwise_treeheap_distance(codes)
        flat_codes = codes.reshape(codes.shape[0], -1)
    else:
        codes = model.codes()
        distance = torch.cdist(codes, codes, p=2)
        flat_codes = codes
    return flat_codes, distance


def context_logits(model, architecture):
    return model.context_logits(temperature=0.35) if architecture == "treeheap" else model.context_logits()


def objective(model, architecture, train_target, echo_weight):
    logits = context_logits(model, architecture)
    positive = train_target > 0.5
    positive_loss = F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive]))
    density_loss = (torch.sigmoid(logits).mean() - train_target.mean()).square()
    context = positive_loss + 4.0 * density_loss
    ids = torch.arange(train_target.shape[1], device=train_target.device)
    echo = F.cross_entropy(model.echo_logits(), ids)
    return context + echo_weight * echo, context, echo


def pair_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    wins = (positive[:, None] < negative[None, :]).float()
    ties = (positive[:, None] == negative[None, :]).float()
    return float((wins + 0.5 * ties).mean().item())


@torch.no_grad()
def measure(model, architecture, train_target, full_target, heldout, object_cat, positive_mask):
    codes, distance = codes_and_distance(model, architecture)
    diagonal = torch.eye(distance.shape[0], dtype=torch.bool, device=distance.device)
    negative_mask = ~positive_mask & ~diagonal
    positives = distance[positive_mask]
    negatives = distance[negative_mask]
    pooled_std = torch.sqrt(0.5 * (positives.var(unbiased=False) + negatives.var(unbiased=False))).clamp_min(1e-8)
    nearest_distance = distance.masked_fill(diagonal, torch.inf)
    nearest = nearest_distance.argmin(-1).cpu().tolist()
    nn_category = sum(object_cat[i] == object_cat[j] for i, j in enumerate(nearest)) / len(object_cat)
    logits = context_logits(model, architecture).cpu()
    transfer = observer.heldout_transfer_metrics(logits, heldout, object_cat)
    ids = torch.arange(train_target.shape[1], device=train_target.device)
    return {
        "pair_distance_auc": pair_auc(positives, negatives),
        "standardized_margin": float(((negatives.mean() - positives.mean()) / pooled_std).item()),
        "positive_distance_native_units": float(positives.mean().item()),
        "negative_distance_native_units": float(negatives.mean().item()),
        "nearest_neighbor_category_accuracy_audit": nn_category,
        "echo_accuracy": float(model.echo_logits().argmax(-1).eq(ids).float().mean().item()),
        "full_context_cell_accuracy": float(((torch.sigmoid(logits) >= 0.5).float() == full_target.cpu()).float().mean().item()),
        "code_variance": float(codes.var(dim=0, unbiased=False).mean().item()),
        **transfer,
    }


def train_architecture(architecture, seed, args, train_target, full_target, heldout, object_cat, positive_mask):
    torch.manual_seed(seed); random.seed(seed)
    if architecture == "treeheap":
        model = observer.MinimalTreeHeapObserver(
            n_verb=train_target.shape[0], n_obj=train_target.shape[1],
            k=args.prefix_slots, dim=args.dim,
        ).to(args.device)
    else:
        model = MetricMLP(
            n_verb=train_target.shape[0], n_obj=train_target.shape[1],
            dim=args.dim, hidden=args.mlp_hidden,
        ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for _ in range(args.base_epochs):
        loss, _, _ = objective(model, architecture, train_target, args.echo_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    before = measure(model, architecture, train_target, full_target, heldout, object_cat, positive_mask)

    for _ in range(args.metric_epochs):
        base_loss, _, _ = objective(model, architecture, train_target, args.echo_weight)
        _, distance = codes_and_distance(model, architecture)
        nce = tree_metric.contrastive_loss(distance, positive_mask, args.temperature)
        loss = base_loss + args.contrastive_weight * nce
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    after = measure(model, architecture, train_target, full_target, heldout, object_cat, positive_mask)
    delta = {key: after[key] - before[key] for key in before}
    return {
        "architecture": architecture, "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "before": before, "after": after, "delta": delta,
    }


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def aggregate(rows):
    output = {}
    for architecture in ("treeheap", "mlp"):
        selected = [row for row in rows if row["architecture"] == architecture]
        metrics = selected[0]["before"].keys()
        output[architecture] = {
            "parameters": selected[0]["parameters"],
            "before": {key: mean_std([row["before"][key] for row in selected]) for key in metrics},
            "after": {key: mean_std([row["after"][key] for row in selected]) for key in metrics},
            "delta": {key: mean_std([row["delta"][key] for row in selected]) for key in metrics},
        }
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_treeheap_vs_mlp_metric_objective")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base-epochs", type=int, default=800)
    ap.add_argument("--metric-epochs", type=int, default=400)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--prefix-slots", type=int, default=6)
    ap.add_argument("--mlp-hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--echo-weight", type=float, default=0.1)
    ap.add_argument("--contrastive-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--seed-start", type=int, default=81)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    started = time.time()
    _, _, verb_to_id, obj_to_id, object_cat = observer.build_vocab()
    structured_train, full_target, heldout = observer.make_structured_pairs(verb_to_id, obj_to_id)
    train_target = structured_train.to(args.device)
    full_target = full_target.to(args.device)
    positive_mask = (train_target.t() @ train_target).gt(0)
    positive_mask.fill_diagonal_(False)
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for architecture in ("treeheap", "mlp"):
            row = train_architecture(architecture, seed, args, train_target, full_target, heldout, object_cat, positive_mask)
            rows.append(row)
            print(json.dumps({
                "seed": seed, "architecture": architecture, "parameters": row["parameters"],
                "auc_after": row["after"]["pair_distance_auc"],
                "mrr_after": row["after"]["heldout_mrr"],
                "auc_delta": row["delta"]["pair_distance_auc"],
            }), flush=True)
    results = aggregate(rows)
    summary = {
        "claim": "S1-METRIC-ARCH-C01", "host": socket.gethostname(),
        "seconds": time.time() - started, "config": vars(args),
        "architectural_contract": {
            "shared": ["corpus", "positive/negative relation", "echo/context/pull-push goal", "epochs", "latent dim"],
            "treeheap": "prefix placement + compose + 3-node depth-weighted diff",
            "mlp": "dense nonlinear latent vector + Euclidean distance",
        },
        "results": results, "rows": rows,
        "scope": "Shared-goal architecture comparison; raw distance units are not compared.",
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
    (out / "README.md").write_text("# TreeHeap vs MLP Under a Shared Metric Goal\n\nSee `summary.json`.\n", encoding="utf8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
