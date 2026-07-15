#!/usr/bin/env python3
"""Self-comparison: add TreeHeap contrastive distance to one observer model."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import s1_encoder_minimal_observer_probe as observer


def treeheap_codes(model: observer.MinimalTreeHeapObserver) -> torch.Tensor:
    prefix, placement = model.prefix_states(temperature=0.35)
    root = placement @ prefix
    leaf = model.object_leaf
    address = placement @ model.prefix_slot
    codes = torch.stack([root, leaf, address], dim=1)
    return F.normalize(codes, dim=-1, eps=1e-8)


def pairwise_treeheap_distance(codes: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([1.0, 0.72, 0.72], device=codes.device, dtype=codes.dtype)
    diff = codes[:, None] - codes[None, :]
    squared = (diff.square().sum(-1) * weights[None, None, :]).sum(-1)
    return torch.sqrt(squared + 1e-8)


def contrastive_loss(distance: torch.Tensor, positive_mask: torch.Tensor, tau: float) -> torch.Tensor:
    n = distance.shape[0]
    diagonal = torch.eye(n, dtype=torch.bool, device=distance.device)
    logits = -distance / tau
    all_logits = logits.masked_fill(diagonal, -torch.inf)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return (torch.logsumexp(all_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def base_loss(model, train_target, echo_weight):
    logits = model.context_logits(temperature=0.35)
    positive = train_target > 0.5
    positive_loss = F.binary_cross_entropy_with_logits(logits[positive], torch.ones_like(logits[positive]))
    density_loss = (torch.sigmoid(logits).mean() - train_target.mean()).square()
    context = positive_loss + 4.0 * density_loss
    ids = torch.arange(model.n_obj, device=train_target.device)
    echo = F.cross_entropy(model.echo_logits(), ids)
    return context + echo_weight * echo, context, echo


@torch.no_grad()
def measure(model, train_target, full_target, heldout, object_cat, positive_mask):
    codes = treeheap_codes(model)
    distance = pairwise_treeheap_distance(codes)
    diagonal = torch.eye(distance.shape[0], dtype=torch.bool, device=distance.device)
    negative_mask = ~positive_mask & ~diagonal
    positive_distance = float(distance[positive_mask].mean().item())
    negative_distance = float(distance[negative_mask].mean().item())
    placement = model.placement(temperature=0.35)
    assignments = placement.argmax(-1).cpu().tolist()
    logits = model.context_logits(temperature=0.35).cpu()
    transfer = observer.heldout_transfer_metrics(logits, heldout, object_cat)
    full_pred = (torch.sigmoid(logits) >= 0.5).float()
    ids = torch.arange(model.n_obj, device=model.object_leaf.device)
    return {
        "positive_distance": positive_distance,
        "negative_distance": negative_distance,
        "margin_negative_minus_positive": negative_distance - positive_distance,
        "echo_accuracy": float(model.echo_logits().argmax(-1).eq(ids).float().mean().item()),
        "full_context_cell_accuracy": float(full_pred.eq(full_target.cpu()).float().mean().item()),
        "cluster_purity_audit": observer.cluster_purity(assignments, object_cat),
        "pairwise_f1_audit": observer.pairwise_f1(assignments, object_cat),
        "effective_prefix_slots": len(set(assignments)),
        "placement_entropy": float((-(placement * placement.clamp_min(1e-9).log()).sum(-1)).mean().item()),
        "code_variance": float(codes.reshape(codes.shape[0], -1).var(dim=0, unbiased=False).mean().item()),
        **transfer,
    }


def train_seed(seed, args, structured_train, full_target, heldout, object_cat):
    torch.manual_seed(seed); random.seed(seed)
    device = torch.device(args.device)
    train_target = structured_train.to(device)
    full_target_device = full_target.to(device)
    model = observer.MinimalTreeHeapObserver(
        n_verb=train_target.shape[0], n_obj=train_target.shape[1],
        k=args.prefix_slots, dim=args.dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    shared_context = train_target.t() @ train_target
    positive_mask = shared_context.gt(0)
    positive_mask.fill_diagonal_(False)

    stage_trace = []
    for epoch in range(1, args.base_epochs + 1):
        loss, context, echo = base_loss(model, train_target, args.echo_weight)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch == args.base_epochs:
            stage_trace.append({"stage": "base", "epoch": epoch, "loss": float(loss.item()), "context": float(context.item()), "echo": float(echo.item())})
    before = measure(model, train_target, full_target_device, heldout, object_cat, positive_mask)

    for epoch in range(1, args.metric_epochs + 1):
        base_total, context, echo = base_loss(model, train_target, args.echo_weight)
        codes = treeheap_codes(model)
        nce = contrastive_loss(pairwise_treeheap_distance(codes), positive_mask, args.temperature)
        loss = base_total + args.contrastive_weight * nce
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch == args.metric_epochs:
            stage_trace.append({"stage": "metric", "epoch": epoch, "loss": float(loss.item()), "context": float(context.item()), "echo": float(echo.item()), "nce": float(nce.item())})
    after = measure(model, train_target, full_target_device, heldout, object_cat, positive_mask)

    keys = [
        "positive_distance", "negative_distance", "margin_negative_minus_positive",
        "echo_accuracy", "heldout_mrr", "heldout_top3", "heldout_beats_other_category",
        "cluster_purity_audit", "pairwise_f1_audit", "code_variance",
        "effective_prefix_slots", "placement_entropy", "full_context_cell_accuracy",
    ]
    delta = {key: after[key] - before[key] for key in keys}
    gates = {
        "margin_gain": delta["margin_negative_minus_positive"] > 0.05,
        "echo_preserved": after["echo_accuracy"] == 1.0,
        "heldout_mrr_preserved": after["heldout_mrr"] >= before["heldout_mrr"] - 0.02,
        "code_not_collapsed": after["code_variance"] > 1e-4,
        "prefix_not_collapsed": after["effective_prefix_slots"] >= 2,
    }
    return {"seed": seed, "before": before, "after": after, "delta": delta, "gates": gates, "trace": stage_trace}


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_treeheap_metric_self_improvement")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base-epochs", type=int, default=800)
    ap.add_argument("--metric-epochs", type=int, default=400)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--prefix-slots", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--echo-weight", type=float, default=0.1)
    ap.add_argument("--contrastive-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--seed-start", type=int, default=71)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    started = time.time()
    _, _, verb_to_id, obj_to_id, object_cat = observer.build_vocab()
    structured_train, full_target, heldout = observer.make_structured_pairs(verb_to_id, obj_to_id)
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        row = train_seed(seed, args, structured_train, full_target, heldout, object_cat)
        rows.append(row)
        print(json.dumps({"seed": seed, "delta": row["delta"], "gates": row["gates"]}), flush=True)

    metrics = list(rows[0]["delta"].keys())
    aggregate = {
        "before": {key: mean_std([row["before"][key] for row in rows]) for key in metrics},
        "after": {key: mean_std([row["after"][key] for row in rows]) for key in metrics},
        "delta": {key: mean_std([row["delta"][key] for row in rows]) for key in metrics},
    }
    gates = {
        "mean_margin_gain": aggregate["delta"]["margin_negative_minus_positive"]["mean"] > 0.05,
        "all_echo_preserved": all(row["gates"]["echo_preserved"] for row in rows),
        "mean_heldout_mrr_preserved": aggregate["delta"]["heldout_mrr"]["mean"] >= -0.02,
        "all_codes_noncollapsed": all(row["gates"]["code_not_collapsed"] for row in rows),
        "all_prefixes_noncollapsed": all(row["gates"]["prefix_not_collapsed"] for row in rows),
    }
    summary = {
        "claim": "S1-SELF-METRIC-C01", "host": socket.gethostname(),
        "seconds": time.time() - started, "config": vars(args),
        "gates": gates, "all_gates_pass": all(gates.values()),
        "aggregate": aggregate, "rows": rows,
        "scope": "TreeHeap self-comparison only; no external superiority claim.",
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
    (out / "README.md").write_text("# TreeHeap Metric Self-Improvement\n\nSee `summary.json`.\n", encoding="utf8")
    print(json.dumps({"gates": gates, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
