#!/usr/bin/env python3
"""Map operator-specific multiscale differences in a frozen TreeHeap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


OPS = ("write_one", "write_two", "mirror_8", "swap_children_8", "swap_subheaps_4", "shuffle_8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def perturb(tokens: torch.Tensor, donor: torch.Tensor, op: str, address: str) -> torch.Tensor:
    out = tokens.clone()
    # A and B use disjoint aligned addresses but identical operation geometry.
    base = 8 if address == "A" else 40
    if op == "write_one":
        out[:, base + 1] = donor[:, base + 1]
    elif op == "write_two":
        out[:, base + 1] = donor[:, base + 1]
        out[:, (base + 25) % 64] = donor[:, (base + 25) % 64]
    elif op == "mirror_8":
        out[:, base : base + 8] = tokens[:, base : base + 8].flip(1)
    elif op == "swap_children_8":
        out[:, base : base + 4] = tokens[:, base + 4 : base + 8]
        out[:, base + 4 : base + 8] = tokens[:, base : base + 4]
    elif op == "swap_subheaps_4":
        other = 48 if address == "A" else 16
        out[:, base : base + 4] = tokens[:, other : other + 4]
        out[:, other : other + 4] = tokens[:, base : base + 4]
    elif op == "shuffle_8":
        # Fixed non-involutive permutation, identical across contents.
        order = torch.tensor((2, 5, 0, 7, 3, 1, 6, 4), device=tokens.device)
        out[:, base : base + 8] = tokens[:, base : base + 8][:, order]
    else:
        raise ValueError(op)
    return out


def inverse(tokens: torch.Tensor, changed: torch.Tensor, op: str, address: str) -> torch.Tensor:
    out = changed.clone()
    base = 8 if address == "A" else 40
    if op == "write_one":
        out[:, base + 1] = tokens[:, base + 1]
        return out
    if op == "write_two":
        out[:, base + 1] = tokens[:, base + 1]
        out[:, (base + 25) % 64] = tokens[:, (base + 25) % 64]
        return out
    if op == "shuffle_8":
        order = torch.tensor((2, 5, 0, 7, 3, 1, 6, 4), device=tokens.device)
        inverse_order = torch.argsort(order)
        out[:, base : base + 8] = changed[:, base : base + 8][:, inverse_order]
        return out
    # Mirror and both swaps are self-inverse.
    return perturb(changed, tokens, op, address)


@torch.no_grad()
def signatures(model, original: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    before = model.levels(original)
    after = model.levels(changed)
    features = []
    for left, right in zip(before, after):
        diff = right - left
        node_energy = diff.square().mean(-1)
        state_cos_change = 1.0 - F.cosine_similarity(left, right, dim=-1)
        bag_change = (model.bag_read(right) - model.bag_read(left)).square().mean(-1)
        adj_change = (model.adj_read(right) - model.adj_read(left)).square().mean(-1)
        features.extend((
            node_energy.mean(1),
            node_energy.max(1).values,
            state_cos_change.mean(1),
            bag_change.mean(1),
            adj_change.mean(1),
        ))
    return torch.stack(features, dim=-1)


def nearest_centroid(train: Dict[str, torch.Tensor], test: Dict[str, torch.Tensor], names=OPS) -> float:
    centroids = torch.stack([train[name].mean(0) for name in names])
    mean = torch.cat([train[name] for name in names]).mean(0)
    std = torch.cat([train[name] for name in names]).std(0).clamp_min(1e-8)
    centroids = F.normalize((centroids - mean) / std, dim=-1)
    correct = total = 0
    for label, name in enumerate(names):
        value = F.normalize((test[name] - mean) / std, dim=-1)
        pred = (value @ centroids.T).argmax(-1)
        correct += int(pred.eq(label).sum())
        total += pred.numel()
    return correct / max(1, total)


def pair_accuracy(train, test, left: str, right: str) -> float:
    return nearest_centroid(train, test, (left, right))


def classification_details(train: Dict[str, torch.Tensor], test: Dict[str, torch.Tensor]) -> dict:
    joined = torch.cat([train[name] for name in OPS])
    mean, std = joined.mean(0), joined.std(0).clamp_min(1e-8)
    centroids = F.normalize(torch.stack([train[name].mean(0) for name in OPS]).sub(mean).div(std), dim=-1)
    confusion = torch.zeros(len(OPS), len(OPS), dtype=torch.int64)
    for label, name in enumerate(OPS):
        value = F.normalize((test[name] - mean) / std, dim=-1)
        pred = (value @ centroids.T).argmax(-1).cpu()
        confusion[label] = torch.bincount(pred, minlength=len(OPS))
    row_sum = confusion.sum(1, keepdim=True).clamp_min(1)
    normalized = confusion.float() / row_sum
    return {
        "labels": list(OPS),
        "per_operator_accuracy": {name: float(normalized[i, i]) for i, name in enumerate(OPS)},
        "row_normalized_confusion": [[round(float(x), 4) for x in row] for row in normalized],
    }


def bootstrap_accuracy(train, test, seed: int, repeats: int = 200) -> List[float]:
    rng = np.random.default_rng(seed)
    values = []
    size = next(iter(test.values())).shape[0]
    for _ in range(repeats):
        index = torch.from_numpy(rng.integers(0, size, size=size))
        values.append(nearest_centroid(train, {name: rows[index] for name, rows in test.items()}))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bag_adj_order_ratio(signature: torch.Tensor) -> tuple[float, float]:
    # Feature layout per depth: energy, max, cosine change, Bag change, Adj change.
    bag = signature[:, 3::5].mean().item()
    adj = signature[:, 4::5].mean().item()
    return bag, adj


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--geometry-script", default=str(here / "s3_treeheap_multiscale_geometry.py"))
    parser.add_argument("--base-script", default=str(here / "s3_residual_treeheap_forest_pretrain.py"))
    parser.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_treeheap_multiscale_geometry_smoke/checkpoint.pt")
    parser.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_treeheap_structural_differential_atlas")
    parser.add_argument("--max-blocks", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=71502)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    geometry = load_module("diff_geometry", Path(args.geometry_script))
    base = load_module("diff_data", Path(args.base_script))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    manifest = base.manifest(Path(args.block_dir), "valid")
    vocab = int(manifest["tokenizer"]["vocab"])
    model = geometry.MultiscaleTreeHeap(vocab, config["dim"], config["sketch_dim"], config["depths"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    collected = {address: {op: [] for op in OPS} for address in ("A", "B")}
    inverse_max_error = 0.0
    seen = 0
    started = time.time()
    for tokens, _ in geometry.batches(base, Path(args.block_dir), "valid", args.batch, args.seed, args.max_blocks + 1):
        tokens = tokens.to(device)
        if seen + tokens.shape[0] > args.max_blocks:
            tokens = tokens[: args.max_blocks - seen]
        if tokens.shape[0] < 2:
            break
        donor = tokens.roll(1, 0)
        for address in ("A", "B"):
            for op in OPS:
                changed = perturb(tokens, donor, op, address)
                collected[address][op].append(signatures(model, tokens, changed).cpu())
                restored = inverse(tokens, changed, op, address)
                with torch.no_grad():
                    restored_states = model.levels(restored)
                    original_states = model.levels(tokens)
                inverse_max_error = max(inverse_max_error, max(float((a - b).abs().max()) for a, b in zip(restored_states, original_states)))
        seen += tokens.shape[0]
        if seen % 2048 == 0:
            print(json.dumps({"blocks": seen, "elapsed_sec": time.time() - started}), flush=True)
        if seen >= args.max_blocks:
            break

    data = {address: {op: torch.cat(rows) for op, rows in ops.items()} for address, ops in collected.items()}
    scales = [size for size in (512, 2048, 8192, 32768) if size <= seen]
    scale_rows = []
    for size in scales:
        train = {op: data["A"][op][:size] for op in OPS}
        test = {op: data["B"][op][:size] for op in OPS}
        row = {
            "blocks": size,
            "multiclass_accuracy": nearest_centroid(train, test),
            "multiclass_bootstrap_95": bootstrap_accuracy(train, test, args.seed + size),
            "mirror_vs_child_swap": pair_accuracy(train, test, "mirror_8", "swap_children_8"),
            "write_one_vs_two": pair_accuracy(train, test, "write_one", "write_two"),
        }
        scale_rows.append(row)

    order_changes = {}
    for op in ("mirror_8", "swap_children_8", "shuffle_8"):
        bag, adj = bag_adj_order_ratio(data["B"][op])
        order_changes[op] = {"bag_read_mse": bag, "adjacency_read_mse": adj, "bag_lt_adjacency": bag < adj}

    final = scale_rows[-1]
    full_train = {op: data["A"][op] for op in OPS}
    full_test = {op: data["B"][op] for op in OPS}
    details = classification_details(full_train, full_test)
    depth_energy = {
        op: [float(x) for x in data["B"][op][:, 0::5].mean(0)]
        for op in OPS
    }
    stable = len(scale_rows) >= 2 and all(
        abs(scale_rows[-1][key] - scale_rows[-2][key]) <= 0.02
        for key in ("multiclass_accuracy", "mirror_vs_child_swap", "write_one_vs_two")
    )
    gates = {
        "P1_inverse": inverse_max_error <= 1e-6,
        "P2_multiclass": final["multiclass_accuracy"] >= (1 / 6 + 0.20),
        "P3_mirror_vs_child": final["mirror_vs_child_swap"] >= 0.60,
        "P4_write_extent": final["write_one_vs_two"] >= 0.70,
        "P5_scale_stable": stable,
        "P6_order_readout": all(row["bag_lt_adjacency"] for row in order_changes.values()),
    }
    if all(gates.values()):
        status = "supported_single_checkpoint"
    elif gates["P1_inverse"] and gates["P2_multiclass"]:
        status = "partial_operator_signatures"
    else:
        status = "not_supported"
    summary = {
        "claim": "S3-TREEHEAP-DIFF-ATLAS-C01",
        "host": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": args.checkpoint,
        "checkpoint_train_blocks": config["train_blocks"],
        "validation_available": manifest["total_blocks"],
        "validation_used": seen,
        "operators": list(OPS),
        "signature_dimensions": 6 * 5,
        "inverse_state_max_abs_error": inverse_max_error,
        "scale_curve": scale_rows,
        "classification": details,
        "operator_depth_mean_state_mse": depth_energy,
        "order_readout_changes": order_changes,
        "gates": gates,
        "status": status,
        "elapsed_sec": time.time() - started,
        "boundary": "Frozen-checkpoint operator response only; no semantic or decoder claim.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
