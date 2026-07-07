#!/usr/bin/env python3
"""Minimal TreeHeap encoder-as-observer proof.

This probe tests the smallest version of the S1 encoder claim:

  L = L_echo + L_context

No hand-provided semantic prefix is used during training.  Gold classes are used
only after training to audit whether the learned TreeHeap placement produced
category-like internal nodes.

The key model object is not a flat pair table.  It has two TreeHeap-shaped
learned parts:

  Theta_place:   object leaf -> soft internal prefix slot
  Theta_compose: objects assigned to one slot -> prefix/node state

    Context prediction then reads through these learned prefix states.  The
    loss treats observed pairs as positives and constrains global score density;
    unobserved held-out pairs are not trained as negatives, otherwise transfer
    would be punished by construction.  A shuffled corpus control keeps token
    counts but destroys the category co-occurrence law.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


Category = Tuple[str, List[str], List[str]]


CATEGORIES: List[Category] = [
    ("food", ["eat", "cook", "order"], ["rice", "noodle", "apple"]),
    ("medicine", ["take", "prescribe", "buy"], ["amoxicillin", "ibuprofen", "aspirin"]),
    ("beverage", ["drink", "pour", "serve"], ["water", "milk", "tea"]),
    ("clothing", ["wear", "wash", "fold"], ["shirt", "hoodie", "coat"]),
    ("vehicle", ["drive", "park", "repair"], ["car", "truck", "bike"]),
    ("place", ["visit", "leave", "enter"], ["paris", "museum", "school"]),
]


@dataclass
class RunConfig:
    seeds: int
    epochs: int
    dim: int
    k_values: List[int]
    lr: float
    echo_weight: float
    device: str
    report_every: int


def build_vocab() -> Tuple[List[str], List[str], Dict[str, int], Dict[str, int], List[int]]:
    verbs: List[str] = []
    objects: List[str] = []
    object_cat: List[int] = []
    for ci, (_cat, cat_verbs, cat_objects) in enumerate(CATEGORIES):
        verbs.extend(cat_verbs)
        for obj in cat_objects:
            objects.append(obj)
            object_cat.append(ci)
    verb_to_id = {v: i for i, v in enumerate(verbs)}
    obj_to_id = {o: i for i, o in enumerate(objects)}
    return verbs, objects, verb_to_id, obj_to_id, object_cat


def make_structured_pairs(
    verb_to_id: Dict[str, int],
    obj_to_id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """Return train target matrix, full target matrix, and held-out pairs.

    For every verb we hold out one object from the matching category.  The held
    out object still appears with sibling verbs, so category-level transfer is
    possible without pair memorization.
    """
    n_verb = len(verb_to_id)
    n_obj = len(obj_to_id)
    train = torch.zeros(n_verb, n_obj)
    full = torch.zeros(n_verb, n_obj)
    heldout: List[Tuple[int, int]] = []

    for ci, (_cat, verbs, objects) in enumerate(CATEGORIES):
        for vi, verb in enumerate(verbs):
            v = verb_to_id[verb]
            held_obj = objects[(vi + ci) % len(objects)]
            for obj in objects:
                o = obj_to_id[obj]
                full[v, o] = 1.0
                if obj == held_obj:
                    heldout.append((v, o))
                else:
                    train[v, o] = 1.0
    return train, full, heldout


def make_shuffled_train(
    structured_train: torch.Tensor,
    object_cat: Sequence[int],
    seed: int,
) -> torch.Tensor:
    """Destroy category law while preserving each verb's positive count."""
    rng = random.Random(seed)
    n_verb, n_obj = structured_train.shape
    shuffled = torch.zeros_like(structured_train)
    all_objects = list(range(n_obj))
    for v in range(n_verb):
        count = int(structured_train[v].sum().item())
        true_cats = {object_cat[o] for o in torch.where(structured_train[v] > 0)[0].tolist()}
        # Prefer negatives from other categories so the control really breaks
        # the law, then fall back to all objects if needed.
        pool = [o for o in all_objects if object_cat[o] not in true_cats]
        if len(pool) < count:
            pool = all_objects
        for o in rng.sample(pool, count):
            shuffled[v, o] = 1.0
    return shuffled


class MinimalTreeHeapObserver(nn.Module):
    """A tiny differentiable TreeHeap encoder.

    Object leaves are softly placed under K internal prefix nodes.  Each prefix
    state is composed from the object leaves assigned to that prefix plus a
    learnable slot vector.  Context prediction reads through those prefix states.
    """

    def __init__(self, n_verb: int, n_obj: int, k: int, dim: int):
        super().__init__()
        self.n_verb = n_verb
        self.n_obj = n_obj
        self.k = k
        self.dim = dim
        self.object_leaf = nn.Parameter(torch.randn(n_obj, dim) * 0.08)
        self.verb_state = nn.Parameter(torch.randn(n_verb, dim) * 0.08)
        self.prefix_slot = nn.Parameter(torch.randn(k, dim) * 0.08)
        self.place_logits = nn.Parameter(torch.randn(n_obj, k) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.echo_head = nn.Linear(dim, n_obj)
        self.object_bias = nn.Parameter(torch.zeros(n_obj))

    def placement(self, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(self.place_logits / temperature, dim=-1)

    def prefix_states(self, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.placement(temperature)
        mass = a.sum(dim=0).clamp_min(1e-6)  # [K]
        weighted = a.t() @ self.object_leaf / mass[:, None]
        mass_feature = torch.log1p(mass)[:, None] / math.log1p(self.n_obj)
        prefix_input = torch.cat([self.prefix_slot, weighted, mass_feature], dim=-1)
        return self.compose(prefix_input), a

    def context_logits(self, temperature: float = 1.0) -> torch.Tensor:
        prefix, a = self.prefix_states(temperature)
        verb_prefix = self.verb_state @ prefix.t() / math.sqrt(self.dim)  # [V,K]
        return verb_prefix @ a.t() + self.object_bias[None, :]

    def echo_logits(self) -> torch.Tensor:
        return self.echo_head(self.object_leaf)


def cluster_purity(assignments: Sequence[int], gold: Sequence[int]) -> float:
    total = len(gold)
    good = 0
    for cluster in sorted(set(assignments)):
        members = [i for i, a in enumerate(assignments) if a == cluster]
        if not members:
            continue
        counts: Dict[int, int] = {}
        for i in members:
            counts[gold[i]] = counts.get(gold[i], 0) + 1
        good += max(counts.values())
    return good / total


def pairwise_f1(assignments: Sequence[int], gold: Sequence[int]) -> float:
    tp = fp = fn = 0
    n = len(gold)
    for i in range(n):
        for j in range(i + 1, n):
            same_pred = assignments[i] == assignments[j]
            same_gold = gold[i] == gold[j]
            if same_pred and same_gold:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif (not same_pred) and same_gold:
                fn += 1
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def heldout_transfer_metrics(
    logits: torch.Tensor,
    heldout: Sequence[Tuple[int, int]],
    object_cat: Sequence[int],
) -> Dict[str, float]:
    ranks: List[int] = []
    outranks: List[float] = []
    for v, o in heldout:
        scores = logits[v].detach().cpu()
        order = torch.argsort(scores, descending=True).tolist()
        rank = order.index(o) + 1
        ranks.append(rank)
        negs = [j for j, c in enumerate(object_cat) if c != object_cat[o]]
        outranks.append(float(scores[o] > scores[negs].max()))
    return {
        "heldout_top1": sum(1 for r in ranks if r == 1) / len(ranks),
        "heldout_top3": sum(1 for r in ranks if r <= 3) / len(ranks),
        "heldout_mrr": sum(1.0 / r for r in ranks) / len(ranks),
        "heldout_beats_other_category": sum(outranks) / len(outranks),
        "heldout_mean_rank": sum(ranks) / len(ranks),
    }


def train_one(
    mode: str,
    seed: int,
    k: int,
    cfg: RunConfig,
    structured_train: torch.Tensor,
    full_target: torch.Tensor,
    heldout: Sequence[Tuple[int, int]],
    object_cat: Sequence[int],
) -> Dict[str, object]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(cfg.device)
    train_target = (
        structured_train
        if mode == "structured"
        else make_shuffled_train(structured_train, object_cat, seed + 10000)
    ).to(device)
    full_target = full_target.to(device)
    model = MinimalTreeHeapObserver(
        n_verb=train_target.shape[0],
        n_obj=train_target.shape[1],
        k=k,
        dim=cfg.dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ids = torch.arange(train_target.shape[1], device=device)

    last = {}
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        # Mild annealing turns soft placement into more TreeHeap-like internal
        # slots without using labels.
        temperature = max(0.35, 1.25 * (1.0 - epoch / (cfg.epochs * 1.3)))
        logits = model.context_logits(temperature=temperature)
        pos_mask = train_target > 0.5
        pos_loss = F.binary_cross_entropy_with_logits(
            logits[pos_mask],
            torch.ones_like(logits[pos_mask]),
        )
        score_density = torch.sigmoid(logits).mean()
        target_density = train_target.mean()
        density_loss = (score_density - target_density).pow(2)
        context_loss = pos_loss + 4.0 * density_loss
        echo_loss = F.cross_entropy(model.echo_logits(), ids)
        loss = context_loss + cfg.echo_weight * echo_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch == cfg.epochs or (cfg.report_every and epoch % cfg.report_every == 0):
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                train_acc = (pred == train_target).float().mean().item()
                echo_acc = (model.echo_logits().argmax(dim=-1) == ids).float().mean().item()
                last = {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "context_loss": float(context_loss.item()),
                    "pos_loss": float(pos_loss.item()),
                    "density_loss": float(density_loss.item()),
                    "score_density": float(score_density.item()),
                    "target_density": float(target_density.item()),
                    "echo_loss": float(echo_loss.item()),
                    "train_bce_cell_acc": train_acc,
                    "echo_acc": echo_acc,
                    "temperature": temperature,
                }

    with torch.no_grad():
        final_logits = model.context_logits(temperature=0.35)
        probs = torch.sigmoid(final_logits).cpu()
        a = model.placement(temperature=0.35).detach().cpu()
        assignments = a.argmax(dim=-1).tolist()
        purity = cluster_purity(assignments, object_cat)
        f1 = pairwise_f1(assignments, object_cat)
        transfer = heldout_transfer_metrics(final_logits.cpu(), heldout, object_cat)
        full_pred = (probs >= 0.5).float()
        full_acc = (full_pred == full_target.cpu()).float().mean().item()
        train_pred = (probs >= 0.5).float()
        train_acc = (train_pred == train_target.cpu()).float().mean().item()

    return {
        "mode": mode,
        "seed": seed,
        "k": k,
        "dim": cfg.dim,
        "epochs": cfg.epochs,
        "seconds": time.time() - start,
        "last": last,
        "cluster_purity": purity,
        "pairwise_f1": f1,
        "full_context_cell_acc": full_acc,
        "train_context_cell_acc": train_acc,
        "assignments": assignments,
        "placement": [[round(float(x), 4) for x in row] for row in a.tolist()],
        **transfer,
    }


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": len(values)}


def summarize(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = [
        "cluster_purity",
        "pairwise_f1",
        "heldout_top1",
        "heldout_top3",
        "heldout_mrr",
        "heldout_beats_other_category",
        "full_context_cell_acc",
    ]
    by_key: Dict[str, Dict[str, object]] = {}
    for mode in sorted({str(r["mode"]) for r in results}):
        rows = [r for r in results if r["mode"] == mode]
        by_key[mode] = {m: mean_std([float(r[m]) for r in rows]) for m in metrics}
    gaps = {}
    if "structured" in by_key and "shuffled" in by_key:
        for m in metrics:
            gaps[m] = (
                by_key["structured"][m]["mean"]  # type: ignore[index]
                - by_key["shuffled"][m]["mean"]  # type: ignore[index]
            )
    return {"by_mode": by_key, "structured_minus_shuffled": gaps}


def write_readme(
    evidence_dir: Path,
    cfg: RunConfig,
    summary: Dict[str, object],
    verbs: Sequence[str],
    objects: Sequence[str],
    object_cat: Sequence[int],
    results: Sequence[Dict[str, object]],
) -> None:
    category_names = [c[0] for c in CATEGORIES]
    best = max(
        (r for r in results if r["mode"] == "structured"),
        key=lambda r: float(r["cluster_purity"]) + float(r["heldout_mrr"]),
    )
    lines = [
        "# S1 Encoder Minimal Observer Probe",
        "",
        "This run tests the DS/Houming minimal gate:",
        "",
        "```text",
        "L = L_echo + L_context",
        "Theta_place learns object -> prefix placement.",
        "Theta_compose learns prefix/internal-node state from assigned leaves.",
        "Gold categories are hidden during training and used only for audit.",
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Example Structured Assignment",
        "",
    ]
    assignments = best["assignments"]
    for obj, ci, cluster in zip(objects, object_cat, assignments):  # type: ignore[arg-type]
        lines.append(f"- `{obj}` gold=`{category_names[ci]}` learned_prefix=`{cluster}`")
    lines += [
        "",
        "## Interpretation Gate",
        "",
        "Support requires structured corpus to beat shuffled control on cluster",
        "purity/pairwise-F1 and held-out transfer. If shuffled matches it, the",
        "encoder is fitting frequency noise rather than observation structure.",
        "",
        "This proof does not claim natural-language semantics or WMT translation.",
        "It only checks whether a learnable TreeHeap placement/compose kernel can",
        "induce reusable internal nodes from observation statistics.",
        "",
    ]
    evidence_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_encoder_minimal_observer_probe")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--k-values", default="4,6,8")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--echo-weight", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-every", type=int, default=0)
    args = ap.parse_args()

    cfg = RunConfig(
        seeds=args.seeds,
        epochs=args.epochs,
        dim=args.dim,
        k_values=[int(x.strip()) for x in args.k_values.split(",") if x.strip()],
        lr=args.lr,
        echo_weight=args.echo_weight,
        device=args.device.strip(),
        report_every=args.report_every,
    )
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    verbs, objects, verb_to_id, obj_to_id, object_cat = build_vocab()
    structured_train, full_target, heldout = make_structured_pairs(verb_to_id, obj_to_id)

    results: List[Dict[str, object]] = []
    trace_path = evidence_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace:
        for k in cfg.k_values:
            for seed in range(cfg.seeds):
                for mode in ("structured", "shuffled"):
                    row = train_one(
                        mode=mode,
                        seed=seed,
                        k=k,
                        cfg=cfg,
                        structured_train=structured_train,
                        full_target=full_target,
                        heldout=heldout,
                        object_cat=object_cat,
                    )
                    results.append(row)
                    trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                    trace.flush()
                    print(
                        f"[{mode}] seed={seed:02d} k={k} "
                        f"purity={row['cluster_purity']:.3f} "
                        f"f1={row['pairwise_f1']:.3f} "
                        f"heldout_mrr={row['heldout_mrr']:.3f} "
                        f"beats_other={row['heldout_beats_other_category']:.3f}",
                        flush=True,
                    )

    summary = {
        "claim": "S1-ENCODER-OBS-C01",
        "experiment": "P-S1-ENCODER-OBS01-minimal",
        "config": asdict(cfg),
        "vocab": {
            "verbs": verbs,
            "objects": objects,
            "categories": [c[0] for c in CATEGORIES],
            "object_category": object_cat,
        },
        "summary": summarize(results),
        "decision_hint": {
            "support_if": "structured beats shuffled on cluster_purity, pairwise_f1, and heldout transfer",
            "reject_or_redesign_if": "shuffled matches structured, or heldout transfer does not improve",
        },
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (evidence_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(evidence_dir, cfg, summary, verbs, objects, object_cat, results)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
