"""Deterministic annealed FOLD/UNFOLD token-space probe.

The model has no trainable token embedding. Token coordinates are root-to-leaf
responsibilities induced from empirical token-context distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


EPS = 1e-12


@dataclass
class Config:
    seeds: int
    seed_start: int
    categories: int
    tokens_per_category: int
    contexts_per_category: int
    global_contexts: int
    train_observations: int
    test_observations: int
    depth: int
    temperatures: List[float]
    em_steps: int
    alpha: float
    device: str


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def make_probability_law(cfg: Config) -> Tuple[torch.Tensor, List[int]]:
    n_tokens = cfg.categories * cfg.tokens_per_category
    category_width = cfg.categories * cfg.contexts_per_category
    signature_start = category_width + cfg.global_contexts
    n_contexts = signature_start + n_tokens
    probs = torch.full((n_tokens, n_contexts), 1e-9, dtype=torch.float64)
    labels: List[int] = []
    for token in range(n_tokens):
        category = token // cfg.tokens_per_category
        labels.append(category)
        lo = category * cfg.contexts_per_category
        hi = lo + cfg.contexts_per_category
        probs[token, lo:hi] = 0.74 / cfg.contexts_per_category
        probs[token, category_width:signature_start] = 0.16 / cfg.global_contexts
        probs[token, signature_start + token] = 0.10
    probs /= probs.sum(dim=1, keepdim=True)
    return probs, labels


def sample_counts(probs: torch.Tensor, observations: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = [
        torch.bincount(
            torch.multinomial(row, observations, replacement=True, generator=generator),
            minlength=probs.shape[1],
        )
        for row in probs
    ]
    return torch.stack(rows).to(torch.float64)


def shuffled_counts(counts: torch.Tensor, seed: int) -> torch.Tensor:
    """Preserve every token's count while destroying shared context identity."""
    rng = random.Random(seed)
    out = torch.zeros_like(counts)
    n_contexts = counts.shape[1]
    for token in range(counts.shape[0]):
        permutation = list(range(n_contexts))
        rng.shuffle(permutation)
        out[token] = counts[token, permutation]
    return out


def distributions(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    smooth = counts + alpha
    return smooth / smooth.sum(dim=1, keepdim=True)


def deterministic_axis(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    center = (weights[:, None] * x).sum(dim=0) / weights.sum().clamp_min(EPS)
    centered = (x - center) * weights.sqrt()[:, None]
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    pivot = int(torch.argmax(axis.abs()).item())
    if axis[pivot] < 0:
        axis = -axis
    return axis


def split_node(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    temperatures: Sequence[float],
    em_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    local = x[indices]
    w = weights[indices]
    axis = deterministic_axis(local, w)
    center = (w[:, None] * local).sum(dim=0) / w.sum().clamp_min(EPS)
    projection = (local - center) @ axis
    order = torch.argsort(projection)
    half = max(1, len(indices) // 2)
    initial = torch.zeros(len(indices), dtype=torch.long, device=x.device)
    initial[order[half:]] = 1
    if int(initial.sum().item()) in (0, len(indices)):
        initial[order[-1]] = 1
        initial[order[0]] = 0

    centers = torch.stack(
        [
            (w[initial == side, None] * local[initial == side]).sum(dim=0)
            / w[initial == side].sum().clamp_min(EPS)
            for side in (0, 1)
        ]
    )
    base = ((local - center) ** 2).sum(dim=1).median().clamp_min(1e-6)
    resp = torch.nn.functional.one_hot(initial, num_classes=2).to(local.dtype)
    for ratio in temperatures:
        temperature = base * float(ratio)
        for _ in range(em_steps):
            distance = ((local[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
            mass = (w[:, None] * resp).sum(dim=0).clamp_min(EPS)
            prior = mass / mass.sum()
            logits = -distance / temperature + prior.log()[None, :]
            resp = torch.softmax(logits, dim=1)
            weighted_resp = w[:, None] * resp
            centers = (weighted_resp.t() @ local) / weighted_resp.sum(dim=0)[:, None].clamp_min(EPS)

    hard = resp.argmax(dim=1)
    if int((hard == 0).sum().item()) == 0 or int((hard == 1).sum().item()) == 0:
        hard = initial
        resp = torch.nn.functional.one_hot(hard, num_classes=2).to(local.dtype)
    child_mass = torch.stack([w[hard == side].sum() for side in (0, 1)])
    child_center = torch.stack(
        [
            (w[hard == side, None] * local[hard == side]).sum(dim=0)
            / w[hard == side].sum().clamp_min(EPS)
            for side in (0, 1)
        ]
    )
    folded = (child_mass[:, None] * child_center).sum(dim=0) / child_mass.sum()
    conservation = float((folded - center).abs().max().item())
    return hard, resp, child_center, conservation


def build_tree(
    x: torch.Tensor,
    weights: torch.Tensor,
    depth: int,
    temperatures: Sequence[float],
    em_steps: int,
) -> Dict[str, object]:
    n_tokens = x.shape[0]
    paths = torch.zeros((n_tokens, depth), dtype=torch.long, device=x.device)
    route_probs = torch.full((n_tokens, depth), 0.5, dtype=x.dtype, device=x.device)
    groups = [torch.arange(n_tokens, device=x.device)]
    conservation: List[float] = []
    partitions: List[List[int]] = []
    for level in range(depth):
        next_groups: List[torch.Tensor] = []
        for indices in groups:
            if len(indices) <= 1:
                next_groups.extend([indices, indices.new_empty(0)])
                continue
            hard, resp, _centers, error = split_node(
                x, indices, weights, temperatures, em_steps
            )
            conservation.append(error)
            paths[indices, level] = hard
            route_probs[indices, level] = resp[:, 1]
            next_groups.extend([indices[hard == 0], indices[hard == 1]])
        groups = next_groups
        ids = torch.zeros(n_tokens, dtype=torch.long, device=x.device)
        for group_id, indices in enumerate(groups):
            if len(indices):
                ids[indices] = group_id
        partitions.append(ids.cpu().tolist())
    leaf_ids = torch.zeros(n_tokens, dtype=torch.long, device=x.device)
    for level in range(depth):
        leaf_ids = leaf_ids * 2 + paths[:, level]
    return {
        "leaf_ids": leaf_ids,
        "paths": paths,
        "route_probs": route_probs,
        "partitions": partitions,
        "conservation_max_abs": max(conservation, default=0.0),
    }


def random_tree(n_tokens: int, depth: int, seed: int, device: torch.device) -> Dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    paths = torch.randint(0, 2, (n_tokens, depth), generator=generator).to(device)
    leaf_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
    partitions: List[List[int]] = []
    for level in range(depth):
        leaf_ids = leaf_ids * 2 + paths[:, level]
        partitions.append(leaf_ids.cpu().tolist())
    return {
        "leaf_ids": leaf_ids,
        "paths": paths,
        "route_probs": paths.to(torch.float64),
        "partitions": partitions,
        "conservation_max_abs": 0.0,
    }


def cluster_purity(assignments: Sequence[int], labels: Sequence[int]) -> float:
    by_cluster: Dict[int, Dict[int, int]] = {}
    for cluster, label in zip(assignments, labels):
        counts = by_cluster.setdefault(cluster, {})
        counts[label] = counts.get(label, 0) + 1
    return sum(max(counts.values()) for counts in by_cluster.values()) / len(labels)


def pairwise_f1(assignments: Sequence[int], labels: Sequence[int]) -> float:
    tp = fp = fn = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            same_pred = assignments[i] == assignments[j]
            same_gold = labels[i] == labels[j]
            tp += int(same_pred and same_gold)
            fp += int(same_pred and not same_gold)
            fn += int(not same_pred and same_gold)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(EPS, precision + recall)


def route_knn_top3(route: torch.Tensor, labels: Sequence[int]) -> float:
    distance = torch.cdist(route, route)
    distance.fill_diagonal_(float("inf"))
    neighbors = distance.topk(k=3, largest=False).indices.cpu().tolist()
    return statistics.fmean(
        sum(labels[j] == labels[i] for j in row) / len(row)
        for i, row in enumerate(neighbors)
    )


def leaf_metrics(leaf_ids: torch.Tensor, depth: int) -> Tuple[float, float]:
    counts = torch.bincount(leaf_ids, minlength=2**depth).to(torch.float64)
    occupied = counts[counts > 0]
    utilization = len(occupied) / (2**depth)
    p = occupied / occupied.sum()
    entropy = float((-(p * p.log()).sum() / math.log(2**depth)).item())
    return utilization, entropy


def leaf_context_nll(
    train_counts: torch.Tensor,
    test_counts: torch.Tensor,
    leaf_ids: torch.Tensor,
    alpha: float,
) -> float:
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=train_counts.device)
    total_count = test_counts.sum()
    for leaf in torch.unique(leaf_ids):
        members = leaf_ids == leaf
        prototype = train_counts[members].sum(dim=0) + alpha
        prototype /= prototype.sum()
        total_loss -= (test_counts[members] * prototype.log()[None, :]).sum()
    return float((total_loss / total_count).item())


def reference_nll(train_counts: torch.Tensor, test_counts: torch.Tensor, alpha: float) -> Tuple[float, float]:
    token_prob = distributions(train_counts, alpha)
    token_nll = float((-(test_counts * token_prob.log()).sum() / test_counts.sum()).item())
    global_prob = train_counts.sum(dim=0) + alpha
    global_prob /= global_prob.sum()
    global_nll = float((-(test_counts * global_prob.log()[None, :]).sum() / test_counts.sum()).item())
    return token_nll, global_nll


def co_cluster_f1(a: Sequence[int], b: Sequence[int]) -> float:
    tp = fp = fn = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            pa = a[i] == a[j]
            pb = b[i] == b[j]
            tp += int(pa and pb)
            fp += int(pa and not pb)
            fn += int(not pa and pb)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(EPS, precision + recall)


def route_stability(results: Sequence[Dict[str, object]], mode: str) -> float:
    partitions = [r["partitions"] for r in results if r["mode"] == mode]
    values: List[float] = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            per_depth = [
                co_cluster_f1(partitions[i][d], partitions[j][d])  # type: ignore[index]
                for d in range(len(partitions[i]))  # type: ignore[arg-type]
            ]
            values.append(statistics.fmean(per_depth))
    return statistics.fmean(values) if values else 1.0


def evaluate_arm(
    mode: str,
    train_counts: torch.Tensor,
    test_counts: torch.Tensor,
    labels: Sequence[int],
    cfg: Config,
    seed: int,
) -> Dict[str, object]:
    device = torch.device(cfg.device)
    train_counts = train_counts.to(device)
    test_counts = test_counts.to(device)
    if mode == "shuffled":
        observed = shuffled_counts(train_counts.cpu(), seed + 700_000).to(device)
    else:
        observed = train_counts
    x = distributions(observed, cfg.alpha).sqrt()
    weights = observed.sum(dim=1)
    if mode == "random_route":
        tree = random_tree(len(labels), cfg.depth, seed + 900_000, device)
    else:
        tree = build_tree(x, weights, cfg.depth, cfg.temperatures, cfg.em_steps)
    leaf_ids = tree["leaf_ids"]
    utilization, occupancy_entropy = leaf_metrics(leaf_ids, cfg.depth)  # type: ignore[arg-type]
    token_nll, global_nll = reference_nll(train_counts, test_counts, cfg.alpha)
    return {
        "mode": mode,
        "seed": seed,
        "heldout_context_nll": leaf_context_nll(observed, test_counts, leaf_ids, cfg.alpha),  # type: ignore[arg-type]
        "token_reference_nll": token_nll,
        "global_reference_nll": global_nll,
        "cluster_purity": cluster_purity(leaf_ids.cpu().tolist(), labels),  # type: ignore[union-attr]
        "pairwise_f1": pairwise_f1(leaf_ids.cpu().tolist(), labels),  # type: ignore[union-attr]
        "route_knn_top3": route_knn_top3(tree["route_probs"], labels),  # type: ignore[arg-type]
        "leaf_utilization": utilization,
        "leaf_occupancy_entropy": occupancy_entropy,
        "fold_conservation_max_abs": tree["conservation_max_abs"],
        "leaf_ids": leaf_ids.cpu().tolist(),  # type: ignore[union-attr]
        "paths": tree["paths"].cpu().tolist(),  # type: ignore[union-attr]
        "route_probabilities": tree["route_probs"].cpu().tolist(),  # type: ignore[union-attr]
        "partitions": tree["partitions"],
    }


def summarize(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = [
        "heldout_context_nll",
        "token_reference_nll",
        "global_reference_nll",
        "cluster_purity",
        "pairwise_f1",
        "route_knn_top3",
        "leaf_utilization",
        "leaf_occupancy_entropy",
        "fold_conservation_max_abs",
    ]
    by_mode: Dict[str, Dict[str, object]] = {}
    for mode in ("structured", "shuffled", "random_route"):
        rows = [row for row in results if row["mode"] == mode]
        by_mode[mode] = {
            metric: mean_std([float(row[metric]) for row in rows]) for metric in metrics
        }
        by_mode[mode]["route_stability"] = route_stability(results, mode)
    real = by_mode["structured"]
    shuffled = by_mode["shuffled"]
    random_route = by_mode["random_route"]
    gaps = {
        "purity_vs_shuffled": real["cluster_purity"]["mean"] - shuffled["cluster_purity"]["mean"],  # type: ignore[index]
        "purity_vs_random": real["cluster_purity"]["mean"] - random_route["cluster_purity"]["mean"],  # type: ignore[index]
        "pairwise_f1_vs_shuffled": real["pairwise_f1"]["mean"] - shuffled["pairwise_f1"]["mean"],  # type: ignore[index]
        "nll_vs_shuffled": shuffled["heldout_context_nll"]["mean"] - real["heldout_context_nll"]["mean"],  # type: ignore[index]
        "nll_vs_random": random_route["heldout_context_nll"]["mean"] - real["heldout_context_nll"]["mean"],  # type: ignore[index]
        "stability_vs_shuffled": real["route_stability"] - shuffled["route_stability"],  # type: ignore[operator]
    }
    gates = {
        "purity_vs_shuffled_ge_0_20": gaps["purity_vs_shuffled"] >= 0.20,
        "purity_vs_random_ge_0_20": gaps["purity_vs_random"] >= 0.20,
        "pairwise_f1_vs_shuffled_ge_0_20": gaps["pairwise_f1_vs_shuffled"] >= 0.20,
        "nll_better_than_shuffled": gaps["nll_vs_shuffled"] > 0.0,
        "nll_better_than_random": gaps["nll_vs_random"] > 0.0,
        "stability_better_than_shuffled": gaps["stability_vs_shuffled"] > 0.0,
        "leaf_utilization_ge_0_75": real["leaf_utilization"]["mean"] >= 0.75,  # type: ignore[index,operator]
        "occupancy_entropy_ge_0_80": real["leaf_occupancy_entropy"]["mean"] >= 0.80,  # type: ignore[index,operator]
        "conservation_le_1e_6": real["fold_conservation_max_abs"]["mean"] <= 1e-6,  # type: ignore[index,operator]
    }
    return {
        "by_mode": by_mode,
        "gaps": gaps,
        "gates": gates,
        "pilot_supported": all(gates.values()),
    }


def write_readme(path: Path, cfg: Config, summary: Dict[str, object], elapsed: float) -> None:
    by_mode = summary["by_mode"]
    lines = [
        "# S1 Annealed FOLD/UNFOLD Token Space",
        "",
        "Claim: `S1-ANNEAL-SPACE-C01`",
        "",
        f"Elapsed seconds: `{elapsed:.3f}`",
        "",
        "No trainable token embedding is used. Labels are audit-only.",
        "",
        "## Means",
        "",
        "| arm | heldout NLL | purity | pair F1 | route top3 | utilization | occupancy H | stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("structured", "shuffled", "random_route"):
        m = by_mode[mode]
        lines.append(
            f"| {mode} | {m['heldout_context_nll']['mean']:.6f} | "
            f"{m['cluster_purity']['mean']:.6f} | {m['pairwise_f1']['mean']:.6f} | "
            f"{m['route_knn_top3']['mean']:.6f} | {m['leaf_utilization']['mean']:.6f} | "
            f"{m['leaf_occupancy_entropy']['mean']:.6f} | {m['route_stability']:.6f} |"
        )
    lines.extend(["", "## Gates", ""])
    for key, value in summary["gates"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            f"Decision: `{'supported pilot' if summary['pilot_supported'] else 'not supported / redesign'}`",
            "",
            "The result is limited to a controlled probability-law corpus. It does not prove WMT semantics or generation.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(asdict(cfg), ensure_ascii=True, indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_annealed_fold_unfold_token_space")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=19001)
    ap.add_argument("--categories", type=int, default=8)
    ap.add_argument("--tokens-per-category", type=int, default=8)
    ap.add_argument("--contexts-per-category", type=int, default=8)
    ap.add_argument("--global-contexts", type=int, default=8)
    ap.add_argument("--train-observations", type=int, default=1200)
    ap.add_argument("--test-observations", type=int, default=600)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--temperatures", default="4,2,1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--em-steps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = Config(
        seeds=args.seeds,
        seed_start=args.seed_start,
        categories=args.categories,
        tokens_per_category=args.tokens_per_category,
        contexts_per_category=args.contexts_per_category,
        global_contexts=args.global_contexts,
        train_observations=args.train_observations,
        test_observations=args.test_observations,
        depth=args.depth,
        temperatures=[float(v) for v in args.temperatures.split(",")],
        em_steps=args.em_steps,
        alpha=args.alpha,
        device=args.device,
    )
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    law, labels = make_probability_law(cfg)
    results: List[Dict[str, object]] = []
    trace = evidence_dir / "trace.jsonl"
    with trace.open("w", encoding="utf-8") as handle:
        for offset in range(cfg.seeds):
            seed = cfg.seed_start + offset
            train = sample_counts(law, cfg.train_observations, seed)
            test = sample_counts(law, cfg.test_observations, seed + 100_000)
            for mode in ("structured", "shuffled", "random_route"):
                row = evaluate_arm(mode, train, test, labels, cfg, seed)
                results.append(row)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                print(
                    f"seed={seed} mode={mode} nll={row['heldout_context_nll']:.6f} "
                    f"purity={row['cluster_purity']:.4f} pair_f1={row['pairwise_f1']:.4f}",
                    flush=True,
                )
    summary = summarize(results)
    elapsed = time.time() - start
    payload = {
        "claim": "S1-ANNEAL-SPACE-C01",
        "experiment": "P-S1-ANNEAL-SPACE01",
        "config": asdict(cfg),
        "vocab": {
            "tokens": cfg.categories * cfg.tokens_per_category,
            "contexts": cfg.categories * cfg.contexts_per_category
            + cfg.global_contexts
            + cfg.categories * cfg.tokens_per_category,
            "labels_audit_only": labels,
        },
        "elapsed_seconds": elapsed,
        "summary": summary,
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (evidence_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (evidence_dir / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_readme(evidence_dir / "README.md", cfg, summary, elapsed)
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
