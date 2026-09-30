#!/usr/bin/env python3
"""A19 shared recursive FOLD/READ state-dimension ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-12


def parse_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SharedRecursiveContextCodec(nn.Module):
    """One FOLD and one READ module reused at every binary depth."""

    def __init__(self, dimension: int, levels: int, feature_mean: torch.Tensor,
                 feature_std: torch.Tensor, output_bias: float):
        super().__init__()
        self.dimension = dimension
        self.levels = levels
        self.register_buffer("feature_mean", feature_mean.reshape(1, 1, 2))
        self.register_buffer("feature_std", feature_std.reshape(1, 1, 2))
        self.leaf_encoder = nn.Linear(2, dimension)

        fold_width = 4 * dimension
        self.fold_candidate = nn.Linear(fold_width, dimension)
        self.fold_gate = nn.Linear(fold_width, dimension)
        self.fold_norm = nn.LayerNorm(dimension)

        self.read_candidate = nn.Linear(dimension, 2 * dimension)
        self.read_gate = nn.Linear(dimension, 2 * dimension)
        self.read_norm = nn.LayerNorm(dimension)

        self.output = nn.Linear(dimension, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, output_bias)

    def leaf_state(self, probability: torch.Tensor) -> torch.Tensor:
        features = torch.stack((probability.clamp_min(EPS).sqrt(),
                                probability.clamp_min(EPS).log()), dim=-1)
        features = (features - self.feature_mean) / self.feature_std
        return torch.tanh(self.leaf_encoder(features))

    def fold_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat((left, right, left - right, left * right), dim=-1)
        candidate = torch.tanh(self.fold_candidate(pair))
        gate = torch.sigmoid(self.fold_gate(pair))
        return self.fold_norm(0.5 * (left + right) + gate * candidate)

    def fold(self, probability: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        state = self.leaf_state(probability)
        statistics = [state_statistics(state)]
        for _ in range(self.levels):
            state = self.fold_pair(state[:, 0::2], state[:, 1::2])
            statistics.append(state_statistics(state))
        return state[:, 0], statistics

    def read_once(self, parent: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.read_candidate(parent)).reshape(
            *parent.shape[:-1], 2, self.dimension)
        gate = torch.sigmoid(self.read_gate(parent)).reshape(
            *parent.shape[:-1], 2, self.dimension)
        child = parent.unsqueeze(-2) + gate * candidate
        return self.read_norm(child)

    def decode(self, root: torch.Tensor, depth: int | None = None,
               branchless: bool = False) -> torch.Tensor:
        used_depth = self.levels if depth is None else depth
        state = root[:, None, :]
        for _ in range(used_depth):
            if branchless:
                state = state.repeat_interleave(2, dim=1)
            else:
                child = self.read_once(state)
                state = child.reshape(state.shape[0], -1, self.dimension)
        if used_depth < self.levels:
            state = state.repeat_interleave(2 ** (self.levels - used_depth), dim=1)
        return self.output(state).squeeze(-1)

    def forward(self, probability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                            List[Dict[str, float]]]:
        root, statistics = self.fold(probability)
        return self.decode(root), root, statistics


def state_statistics(state: torch.Tensor) -> Dict[str, float]:
    detached = state.detach().to(torch.float64)
    return {
        "nodes": int(detached.shape[1]),
        "rms": float(detached.square().mean().sqrt().cpu()),
        "dimension_variance": float(detached.var(dim=(0, 1), unbiased=False).mean().cpu()),
        "sample_variance": float(detached.var(dim=0, unbiased=False).mean().cpu()),
    }


def pair_nll(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return -(counts * F.log_softmax(logits, dim=-1)).sum() / counts.sum().clamp_min(1.0)


def evaluate_logits(logits: torch.Tensor, counts: torch.Tensor) -> Dict[str, float | bool]:
    with torch.no_grad():
        nll = pair_nll(logits, counts)
        top1 = logits.argmax(dim=-1)
        target_top1 = counts.argmax(dim=-1)
        return {
            "nll": float(nll.cpu()),
            "ppl": float(torch.exp(nll.clamp_max(30.0)).cpu()),
            "row_top1": float((top1 == target_top1).to(torch.float32).mean().cpu()),
            "finite": bool(torch.isfinite(logits).all() and torch.isfinite(nll)),
        }


def effective_rank(root: torch.Tensor) -> Dict[str, float]:
    centered = root.detach().to(torch.float64) - root.detach().to(torch.float64).mean(0)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(EPS)
    entropy_rank = torch.exp(-(probability * probability.clamp_min(EPS).log()).sum())
    participation = energy.sum().square() / energy.square().sum().clamp_min(EPS)
    normalized = F.normalize(root.detach().to(torch.float64), dim=-1)
    cosine = normalized @ normalized.t()
    mask = ~torch.eye(len(root), dtype=torch.bool, device=root.device)
    return {
        "numerical_rank": int(torch.linalg.matrix_rank(centered).item()),
        "entropy_effective_rank": float(entropy_rank.cpu()),
        "participation_rank": float(participation.cpu()),
        "mean_offdiag_cosine": float(cosine[mask].mean().cpu()),
    }


def fixed_batches(rows: int, batch_size: int, epochs: int,
                  seed: int) -> Iterable[Tuple[int, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(rows, generator=generator)
        for start in range(0, rows, batch_size):
            yield step, order[start:start + batch_size]
            step += 1


def train_dimension(dimension: int, probability: torch.Tensor, fit_counts: torch.Tensor,
                    dev_counts: torch.Tensor, test_counts: torch.Tensor, levels: int,
                    epochs: int, batch_size: int, learning_rate: float,
                    weight_decay: float, seed: int, eval_epochs: List[int],
                    out: Path) -> Dict[str, object]:
    device = probability.device
    features = torch.stack((probability.clamp_min(EPS).sqrt(),
                            probability.clamp_min(EPS).log()), dim=-1)
    feature_mean = features.mean(dim=(0, 1))
    feature_std = features.std(dim=(0, 1), unbiased=False).clamp_min(1e-5)
    global_probability = fit_counts.sum(dim=0) / fit_counts.sum().clamp_min(1.0)
    output_bias = float(global_probability.clamp_min(EPS).log().mean().cpu())

    torch.manual_seed(seed + dimension * 1009)
    torch.cuda.manual_seed_all(seed + dimension * 1009)
    model = SharedRecursiveContextCodec(
        dimension, levels, feature_mean, feature_std, output_bias).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * math.ceil(len(probability) / batch_size)
    eval_steps = {0, total_steps}
    for epoch in eval_epochs:
        eval_steps.add(epoch * math.ceil(len(probability) / batch_size))
    trajectory: List[Dict[str, object]] = []
    finite = True
    maximum_gradient_norm = 0.0
    started = time.time()

    def snapshot(step: int) -> None:
        model.eval()
        with torch.no_grad():
            logits, root, fold_statistics = model(probability)
            row = {
                "step": step,
                "epoch": step / math.ceil(len(probability) / batch_size),
                "fit": evaluate_logits(logits, fit_counts),
                "dev": evaluate_logits(logits, dev_counts),
                "test": evaluate_logits(logits, test_counts),
                "root": effective_rank(root),
                "fold_levels": fold_statistics,
            }
        trajectory.append(row)
        print(json.dumps({"dimension": dimension, **row}), flush=True)
        model.train()

    snapshot(0)
    optimizer.zero_grad(set_to_none=True)
    for step, batch_cpu in fixed_batches(len(probability), batch_size, epochs, seed + 17):
        batch = batch_cpu.to(device)
        logits, _root, _statistics = model(probability[batch])
        loss = pair_nll(logits, fit_counts[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        finite = finite and bool(torch.isfinite(loss)) and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients)
        gradient_norm = math.sqrt(sum(float(gradient.detach().square().sum().cpu())
                                      for gradient in gradients))
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        if not finite:
            raise RuntimeError(f"non-finite training state at dimension={dimension} step={step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        completed_step = step + 1
        if completed_step in eval_steps:
            snapshot(completed_step)

    model.eval()
    with torch.no_grad():
        final_logits, root, fold_statistics = model(probability)
        depth_curve = []
        for depth in range(levels + 1):
            logits = model.decode(root, depth=depth)
            depth_curve.append({
                "read_depth": depth,
                "distinct_regions": 2**depth,
                "test": evaluate_logits(logits, test_counts),
            })
        generator = torch.Generator(device="cpu").manual_seed(seed + 9000 + dimension)
        permutation = torch.randperm(len(root), generator=generator).to(device)
        shuffled = evaluate_logits(model.decode(root[permutation]), test_counts)
        zero_root = evaluate_logits(model.decode(torch.zeros_like(root)), test_counts)
        branchless = evaluate_logits(model.decode(root, branchless=True), test_counts)
        final = {
            "fit": evaluate_logits(final_logits, fit_counts),
            "dev": evaluate_logits(final_logits, dev_counts),
            "test": evaluate_logits(final_logits, test_counts),
        }

    checkpoint = out / f"dimension_{dimension}.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = SharedRecursiveContextCodec(
        dimension, levels, feature_mean, feature_std, output_bias).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        reload_logits, reload_root, _ = reloaded(probability)
        reload_max_abs = max(
            float((reload_logits - final_logits).abs().max().cpu()),
            float((reload_root - root).abs().max().cpu()),
        )
    result = {
        "dimension": dimension,
        "levels": levels,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "steps": total_steps,
        "finite": finite,
        "maximum_gradient_norm": maximum_gradient_norm,
        "trajectory": trajectory,
        "final": final,
        "depth_curve": depth_curve,
        "controls": {
            "shuffled_root_test": shuffled,
            "zero_root_test": zero_root,
            "branchless_read_test": branchless,
            "shuffle_permutation_sha256": tensor_hash(permutation),
        },
        "root": effective_rank(root),
        "fold_levels": fold_statistics,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "state_sha256": state_hash(model),
            "reload_max_abs": reload_max_abs,
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(out / f"dimension_{dimension}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dimensions", default="2,4,8,16")
    parser.add_argument("--fit-ratio", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--eval-epochs", default="1,5,10,25,50,100,150,200")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts_path = Path(args.counts)
    payload = torch.load(counts_path, map_location="cpu", weights_only=False)
    original_train = payload["train"].to(torch.float64)
    sealed_test = payload["test"].to(torch.float64)
    if original_train.ndim != 2 or original_train.shape[1] & (original_train.shape[1] - 1):
        raise ValueError("context width must be a power of two")
    levels = int(math.log2(original_train.shape[1]))
    generator = torch.Generator(device="cpu").manual_seed(args.split_seed)
    fit = torch.binomial(
        original_train, torch.full_like(original_train, args.fit_ratio), generator=generator)
    dev = original_train - fit
    probability = (fit + args.alpha) / (
        fit.sum(dim=1, keepdim=True) + args.alpha * fit.shape[1])
    device = torch.device(args.device)
    probability = probability.to(device=device, dtype=torch.float32)
    fit = fit.to(device=device, dtype=torch.float32)
    dev = dev.to(device=device, dtype=torch.float32)
    sealed_test = sealed_test.to(device=device, dtype=torch.float32)
    dimensions = parse_ints(args.dimensions)
    eval_epochs = parse_ints(args.eval_epochs)
    source_hash = file_sha256(Path(__file__).resolve())
    material = {
        "claim": "S1-RECURSIVE-CONTEXT-CODEC-A19-C01",
        "boundary": "Shared recursive context-field codec dimension ladder; not sentence generation or translation.",
        "counts": str(counts_path),
        "counts_sha256": file_sha256(counts_path),
        "source_sha256": source_hash,
        "shape": list(original_train.shape),
        "levels": levels,
        "fit_pairs": float(fit.sum().cpu()),
        "dev_pairs": float(dev.sum().cpu()),
        "sealed_test_pairs": float(sealed_test.sum().cpu()),
        "target_ids_sha256": hashlib.sha256(json.dumps(
            payload.get("target_ids", []), separators=(",", ":")).encode()).hexdigest(),
        "context_ids_sha256": hashlib.sha256(json.dumps(
            payload.get("context_ids", []), separators=(",", ":")).encode()).hexdigest(),
        "config": vars(args),
    }
    write_json(out / "material.json", material)

    results = []
    for dimension in dimensions:
        results.append(train_dimension(
            dimension, probability, fit, dev, sealed_test, levels,
            args.epochs, args.batch_size, args.learning_rate,
            args.weight_decay, args.seed, eval_epochs, out))

    rows = []
    for result in results:
        test_nll = float(result["final"]["test"]["nll"])
        depth_curve = result["depth_curve"]
        read_gain = float(depth_curve[0]["test"]["nll"]) - test_nll
        shuffle_damage = float(result["controls"]["shuffled_root_test"]["nll"]) - test_nll
        branchless_damage = float(result["controls"]["branchless_read_test"]["nll"]) - test_nll
        rows.append({
            "dimension": result["dimension"],
            "parameters": result["parameters"],
            "test_nll": test_nll,
            "test_ppl": result["final"]["test"]["ppl"],
            "read_depth_gain": read_gain,
            "shuffle_damage": shuffle_damage,
            "branchless_damage": branchless_damage,
            "root_effective_rank": result["root"]["entropy_effective_rank"],
            "reload_max_abs": result["checkpoint"]["reload_max_abs"],
            "finite": result["finite"],
            "full_budget": result["steps"] == args.epochs * math.ceil(len(probability) / args.batch_size),
        })
    gates = {
        "all_dimensions_completed": len(rows) == len(dimensions),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "all_full_budget": all(bool(row["full_budget"]) for row in rows),
        "all_reload_exact": all(float(row["reload_max_abs"]) == 0.0 for row in rows),
        "one_dimension_beats_shuffled_root": any(float(row["shuffle_damage"]) > 0.01 for row in rows),
        "one_dimension_beats_branchless_read": any(float(row["branchless_damage"]) > 0.01 for row in rows),
        "one_dimension_has_nonflat_depth_curve": any(float(row["read_depth_gain"]) > 0.01 for row in rows),
    }
    summary = {
        "claim": "S1-RECURSIVE-CONTEXT-CODEC-A19-C01",
        "material": material,
        "rows": rows,
        "gates": gates,
        "supported": all(gates.values()),
    }
    write_json(out / "summary.json", summary)
    with (out / "dimension_ladder.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
