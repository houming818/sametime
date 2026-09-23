"""Monte Carlo search for a probability-residual TreeHeap embedding F."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch


EPS = 1e-12


@dataclass
class SearchState:
    axes: torch.Tensor
    thresholds: torch.Tensor
    dev_nll: float


def probability(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    return (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * counts.shape[1])


def route(x: torch.Tensor, axes: torch.Tensor, thresholds: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    nodes = torch.zeros(n, dtype=torch.long, device=x.device)
    visited = []
    rows = torch.arange(n, device=x.device)
    for _ in range(depth):
        visited.append(nodes.clone())
        local_axis = axes[nodes]
        projection = (x * local_axis).sum(dim=1)
        go_right = projection > thresholds[nodes]
        nodes = nodes * 2 + 1 + go_right.long()
    leaves = nodes - (2**depth - 1)
    return leaves, torch.stack(visited, dim=1)


def leaf_probability(fit_counts: torch.Tensor, assignment: torch.Tensor, leaves: int, alpha: float) -> torch.Tensor:
    result = torch.zeros((leaves, fit_counts.shape[1]), dtype=fit_counts.dtype, device=fit_counts.device)
    result.index_add_(0, assignment, fit_counts)
    return probability(result, alpha)


def nll_for_assignment(fit_counts: torch.Tensor, eval_counts: torch.Tensor, assignment: torch.Tensor, leaves: int, alpha: float) -> float:
    leaf_prob = leaf_probability(fit_counts, assignment, leaves, alpha)
    value = -(eval_counts * leaf_prob[assignment].clamp_min(EPS).log()).sum() / eval_counts.sum().clamp_min(1.0)
    return float(value.item())


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    order = torch.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = q * float(ordered_weights.sum().item())
    index = int(
        torch.searchsorted(
            ordered_weights.cumsum(0),
            torch.tensor(cutoff, dtype=ordered_weights.dtype, device=values.device),
        ).item()
    )
    index = min(index, len(ordered_values) - 1)
    return float(ordered_values[index].item())


def deterministic_initial(x: torch.Tensor, weights: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    internal = 2**depth - 1
    axes = torch.zeros((internal, x.shape[1]), dtype=x.dtype, device=x.device)
    thresholds = torch.zeros(internal, dtype=x.dtype, device=x.device)
    members: Dict[int, torch.Tensor] = {0: torch.arange(x.shape[0], device=x.device)}
    fallback = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    fallback[0] = 1.0
    for node in range(internal):
        ids = members.get(node, torch.empty(0, dtype=torch.long, device=x.device))
        if len(ids) < 2:
            axes[node] = fallback
            thresholds[node] = 0.0
            if 2 * node + 1 < internal:
                members[2 * node + 1] = ids
                members[2 * node + 2] = torch.empty(0, dtype=torch.long, device=x.device)
            continue
        local = x[ids]
        local_weight = weights[ids]
        center = (local * local_weight[:, None]).sum(0) / local_weight.sum().clamp_min(EPS)
        first = int(((local - center) ** 2).sum(1).argmax().item())
        second = int(((local - local[first]) ** 2).sum(1).argmax().item())
        axis = local[first] - local[second]
        axis = axis / axis.norm().clamp_min(EPS)
        projection = local @ axis
        threshold = weighted_quantile(projection, local_weight, 0.5)
        axes[node] = axis
        thresholds[node] = threshold
        if 2 * node + 1 < internal:
            right = projection > threshold
            members[2 * node + 1] = ids[~right]
            members[2 * node + 2] = ids[right]
    return axes, thresholds


def random_assignment(n: int, leaves: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(leaves, (n,), generator=generator, device="cpu").to(device)


def utilization_entropy(assignment: torch.Tensor, weights: torch.Tensor, leaves: int) -> Tuple[float, float]:
    mass = torch.zeros(leaves, dtype=weights.dtype, device=weights.device)
    mass.index_add_(0, assignment, weights)
    used = float((mass > 0).float().mean().item())
    p = mass / mass.sum().clamp_min(EPS)
    positive = p > 0
    entropy = float((-(p[positive] * p[positive].log()).sum() / math.log(leaves)).item())
    return used, entropy


def algebra_audit(fit_counts: torch.Tensor, assignment: torch.Tensor, depth: int) -> Dict[str, float]:
    leaves = 2**depth
    total_nodes = 2 ** (depth + 1) - 1
    leaf_offset = leaves - 1
    node_counts = torch.zeros((total_nodes, fit_counts.shape[1]), dtype=fit_counts.dtype, device=fit_counts.device)
    node_mass = torch.zeros(total_nodes, dtype=fit_counts.dtype, device=fit_counts.device)
    row_mass = fit_counts.sum(dim=1)
    for token in range(fit_counts.shape[0]):
        node = leaf_offset + int(assignment[token].item())
        while True:
            node_counts[node] += fit_counts[token]
            node_mass[node] += row_mass[token]
            if node == 0:
                break
            node = (node - 1) // 2
    raw_probability = node_counts / node_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
    conservation = 0.0
    for node in range(leaf_offset):
        left, right = 2 * node + 1, 2 * node + 2
        mass = node_mass[left] + node_mass[right]
        if mass <= 0:
            continue
        folded = (node_mass[left] * raw_probability[left] + node_mass[right] * raw_probability[right]) / mass
        conservation = max(conservation, float((folded - raw_probability[node]).abs().max().item()))
    closure = 0.0
    for leaf in range(leaves):
        node = leaf_offset + leaf
        if node_mass[node] <= 0:
            continue
        path = []
        cursor = node
        while cursor != 0:
            path.append(cursor)
            cursor = (cursor - 1) // 2
        reconstructed = raw_probability[0].clone()
        for child in reversed(path):
            parent = (child - 1) // 2
            reconstructed += raw_probability[child] - raw_probability[parent]
        closure = max(closure, float((reconstructed - raw_probability[node]).abs().max().item()))
    return {"fold_conservation_max_abs": conservation, "residual_closure_max_abs": closure}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_embedding_checkpoint(
    path: Path,
    fit_counts: torch.Tensor,
    x: torch.Tensor,
    assignment: torch.Tensor,
    visited: torch.Tensor,
    axes: torch.Tensor,
    thresholds: torch.Tensor,
    depth: int,
    alpha: float,
    source_payload: Dict[str, object],
    vocab_metadata: Dict[str, object],
    claim: str,
) -> Dict[str, object]:
    leaves = 2**depth
    leaf_offset = leaves - 1
    total_nodes = 2 ** (depth + 1) - 1
    node_counts = torch.zeros((total_nodes, fit_counts.shape[1]), dtype=fit_counts.dtype, device=fit_counts.device)
    node_counts[leaf_offset:].index_add_(0, assignment, fit_counts)
    for node in range(leaf_offset - 1, -1, -1):
        node_counts[node] = node_counts[2 * node + 1] + node_counts[2 * node + 2]
    node_mass = node_counts.sum(dim=1)
    node_probability = node_counts / node_mass[:, None].clamp_min(EPS)
    node_residual = torch.zeros_like(node_probability)
    for node in range(1, total_nodes):
        node_residual[node] = node_probability[node] - node_probability[(node - 1) // 2]
    rows = torch.arange(x.shape[0], device=x.device)[:, None]
    local_axes = axes[visited]
    route_margin = (x[:, None, :] * local_axes).sum(dim=2) - thresholds[visited]
    path_bits = (route_margin > 0).to(torch.uint8)
    leaf_smoothed_probability = probability(node_counts[leaf_offset:], alpha)
    checkpoint = {
        "format": "treeheap_probability_residual_embedding_v1",
        "claim": claim,
        "depth": depth,
        "target_ids": source_payload.get("target_ids"),
        "context_ids": source_payload.get("context_ids"),
        "target_pieces": vocab_metadata.get("target_pieces"),
        "context_pieces": vocab_metadata.get("context_pieces"),
        "spm_sha256": vocab_metadata.get("spm_sha256"),
        "axes": axes.detach().cpu(),
        "thresholds": thresholds.detach().cpu(),
        "token_leaf": assignment.detach().cpu(),
        "token_path_bits": path_bits.detach().cpu(),
        "token_route_margin": route_margin.detach().cpu(),
        "token_context_sqrt_probability": x.detach().cpu(),
        "node_mass": node_mass.detach().cpu(),
        "node_probability": node_probability.detach().cpu(),
        "node_residual": node_residual.detach().cpu(),
        "leaf_smoothed_probability": leaf_smoothed_probability.detach().cpu(),
    }
    torch.save(checkpoint, path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "format": checkpoint["format"],
        "target_tokens": int(x.shape[0]),
        "context_tokens": int(x.shape[1]),
        "leaves": leaves,
        "route_dimensions": depth,
    }


def propose(state: SearchState, x: torch.Tensor, weights: torch.Tensor, depth: int, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    axes = state.axes.clone()
    thresholds = state.thresholds.clone()
    _leaves, visited = route(x, axes, thresholds, depth)
    candidates = []
    for level in range(depth):
        for node in torch.unique(visited[:, level]).tolist():
            ids = torch.where(visited[:, level] == node)[0]
            if len(ids) >= 3:
                candidates.append((int(node), ids))
    node, ids = rng.choice(candidates)
    local = x[ids]
    mode = "pair" if rng.random() < 0.65 else "perturb"
    if mode == "pair":
        a, b = rng.sample(range(len(ids)), 2)
        axis = local[a] - local[b]
    else:
        noise = torch.randn(axes[node].shape, generator=torch.Generator(device="cpu").manual_seed(rng.randrange(2**31)), dtype=axes.dtype, device="cpu").to(x.device)
        axis = axes[node] + rng.choice([0.05, 0.10, 0.20]) * noise / math.sqrt(x.shape[1])
    if float(axis.norm().item()) <= EPS:
        axis = axes[node].clone()
    axis = axis / axis.norm().clamp_min(EPS)
    projection = local @ axis
    q = rng.choice([0.2, 0.35, 0.5, 0.65, 0.8])
    threshold = weighted_quantile(projection, weights[ids], q)
    axes[node] = axis
    thresholds[node] = threshold
    return axes, thresholds, {"node": node, "members": len(ids), "mode": mode, "quantile": q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="ara/s1-echo/evidence/s1_real_corpus_annealed_token_space/formal_200k_seed19101/context_counts.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--fit-ratio", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--search-seed", type=int)
    parser.add_argument("--claim", default="S1-F-MC-A11-C01")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--vocab-json", default="ara/s1-echo/evidence/s1_real_corpus_annealed_token_space/formal_200k_seed19101/vocab.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload = torch.load(args.counts, map_location="cpu", weights_only=False)
    original_train = payload["train"].to(torch.float64)
    sealed_test = payload["test"].to(device=device, dtype=torch.float64)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    search_seed = args.seed if args.search_seed is None else args.search_seed
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    fit = torch.binomial(original_train, torch.full_like(original_train, args.fit_ratio), generator=generator)
    search_dev = original_train - fit
    fit = fit.to(device)
    search_dev = search_dev.to(device)
    x = probability(fit, args.alpha).sqrt()
    weights = fit.sum(dim=1).clamp_min(1.0)
    leaves = 2**args.depth
    initial_axes, initial_thresholds = deterministic_initial(x, weights, args.depth)
    initial_assignment, _ = route(x, initial_axes, initial_thresholds, args.depth)
    initial_dev = nll_for_assignment(fit, search_dev, initial_assignment, leaves, args.alpha)
    initial = SearchState(initial_axes, initial_thresholds, initial_dev)
    current = copy.deepcopy(initial)
    best = copy.deepcopy(initial)
    rng = random.Random(search_seed + 1)
    trace: List[Dict[str, object]] = []
    accepted_count = 0
    for iteration in range(args.iterations):
        axes, thresholds, proposal_meta = propose(current, x, weights, args.depth, rng)
        assignment, _ = route(x, axes, thresholds, args.depth)
        dev_nll = nll_for_assignment(fit, search_dev, assignment, leaves, args.alpha)
        temperature = args.temperature * (0.985**iteration)
        improvement = current.dev_nll - dev_nll
        accepted = improvement >= 0 or rng.random() < math.exp(improvement / max(temperature, 1e-6))
        if accepted:
            current = SearchState(axes, thresholds, dev_nll)
            accepted_count += 1
        if dev_nll < best.dev_nll:
            best = SearchState(axes.clone(), thresholds.clone(), dev_nll)
        row = {
            "iteration": iteration,
            "proposal": proposal_meta,
            "proposal_dev_nll": dev_nll,
            "accepted": accepted,
            "current_dev_nll": current.dev_nll,
            "best_dev_nll": best.dev_nll,
            "temperature": temperature,
        }
        trace.append(row)
        if iteration == 0 or (iteration + 1) % 32 == 0:
            print(json.dumps(row), flush=True)

    best_assignment, best_visited = route(x, best.axes, best.thresholds, args.depth)
    random_route = random_assignment(x.shape[0], leaves, search_seed + 2, device)
    initial_test = nll_for_assignment(fit, sealed_test, initial_assignment, leaves, args.alpha)
    best_test = nll_for_assignment(fit, sealed_test, best_assignment, leaves, args.alpha)
    random_test = nll_for_assignment(fit, sealed_test, random_route, leaves, args.alpha)
    utilization, occupancy_entropy = utilization_entropy(best_assignment, weights, leaves)
    audit = algebra_audit(fit, best_assignment, args.depth)
    finite = all(math.isfinite(value) for value in [initial_dev, best.dev_nll, initial_test, best_test, random_test])
    gates = {
        "finite": finite,
        "accepted_at_least_one": accepted_count > 0,
        "fold_conservation_le_1e_10": audit["fold_conservation_max_abs"] <= 1e-10,
        "residual_closure_le_1e_10": audit["residual_closure_max_abs"] <= 1e-10,
        "dev_gain_ge_0_005": initial_dev - best.dev_nll >= 0.005,
        "sealed_test_not_worse_gt_0_01": best_test <= initial_test + 0.01,
        "sealed_test_better_than_random": best_test < random_test,
        "leaf_utilization_ge_0_50": utilization >= 0.50,
    }
    summary = {
        "claim": args.claim,
        "boundary": "Corpus context-field embedding F search only; no next-token, decoder, READ, translation, or generation target.",
        "config": {**vars(args), "resolved_split_seed": split_seed, "resolved_search_seed": search_seed},
        "shape": {"tokens": fit.shape[0], "contexts": fit.shape[1], "leaves": leaves},
        "counts": {"fit": float(fit.sum().item()), "search_dev": float(search_dev.sum().item()), "sealed_test": float(sealed_test.sum().item())},
        "search": {
            "initial_dev_nll": initial_dev,
            "best_dev_nll": best.dev_nll,
            "dev_gain": initial_dev - best.dev_nll,
            "accepted": accepted_count,
            "iterations": args.iterations,
        },
        "sealed_evaluation": {
            "initial_test_nll": initial_test,
            "best_test_nll": best_test,
            "random_test_nll": random_test,
            "best_minus_initial": best_test - initial_test,
            "best_minus_random": best_test - random_test,
        },
        "structure": {"leaf_utilization": utilization, "occupancy_entropy": occupancy_entropy, **audit},
        "gates": gates,
        "claim_supported": all(gates.values()),
    }
    if args.save_checkpoint:
        vocab_metadata = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))
        checkpoint_path = out / "embedding_checkpoint.pt"
        checkpoint = export_embedding_checkpoint(
            checkpoint_path,
            fit,
            x,
            best_assignment,
            best_visited,
            best.axes,
            best.thresholds,
            args.depth,
            args.alpha,
            payload,
            vocab_metadata,
            args.claim,
        )
        reloaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        reload_assignment, _ = route(x, reloaded["axes"].to(device), reloaded["thresholds"].to(device), args.depth)
        reload_nll = nll_for_assignment(fit, sealed_test, reload_assignment, leaves, args.alpha)
        checkpoint["reload_assignment_exact"] = bool(torch.equal(reload_assignment, best_assignment))
        checkpoint["reload_test_nll_abs_delta"] = abs(reload_nll - best_test)
        checkpoint["reload_pass"] = checkpoint["reload_assignment_exact"] and checkpoint["reload_test_nll_abs_delta"] <= 1e-12
        summary["checkpoint"] = checkpoint
        summary["gates"]["checkpoint_reload_exact"] = checkpoint["reload_pass"]
        summary["claim_supported"] = all(summary["gates"].values())
    (out / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "command.txt").write_text("python3 " + " ".join(__file__ for _ in [0]) + "\n" + json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
