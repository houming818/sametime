#!/usr/bin/env python3
"""Learn operator programs; execute native TreeHeap algebra; test OOD structure."""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


OPS = ["mirror", "plus"]
DELTAS = [-3, -1, 1, 3]
MAX_TREE_DEPTH = 8
MAX_NODES = 2 ** (MAX_TREE_DEPTH + 1) - 1
MAX_OP_DEPTH = 4


def level(address: int) -> int:
    return address.bit_length() - 1


def legal_heap(tree_depth: int, root: int) -> np.ndarray:
    size = 2 ** (tree_depth + 1) - 1
    arr = np.zeros(MAX_NODES + 1, dtype=np.float32)
    arr[1] = root
    for address in range(2, size + 1):
        parent = address // 2
        arr[address] = 2 * arr[parent] + (address & 1)
    return arr


def subtree_addresses(root: int, depth: int, size: int) -> List[int]:
    out = []
    for relative_depth in range(depth + 1):
        start = root * (2 ** relative_depth)
        for offset in range(2 ** relative_depth):
            address = start + offset
            if address <= size:
                out.append(address)
    return out


def mirror_subtree(arr: np.ndarray, root: int, depth: int, size: int) -> np.ndarray:
    out = arr.copy()
    for relative_depth in range(1, depth + 1):
        start = root * (2 ** relative_depth)
        width = 2 ** relative_depth
        original = arr[start:start + width].copy()
        out[start:start + width] = original[::-1]
    return out


def plus_subtree(arr: np.ndarray, root: int, depth: int, delta: int, size: int) -> np.ndarray:
    out = arr.copy()
    for address in subtree_addresses(root, depth, size):
        out[address] += delta
    return out


def execute_inverse(arr: np.ndarray, op: int, address: int, depth: int, delta_index: int, size: int) -> np.ndarray:
    if address < 1 or address > size or depth < 1 or address * (2 ** depth) > size:
        return arr.copy()
    if op == 0:
        return mirror_subtree(arr, address, depth, size)
    return plus_subtree(arr, address, depth, -DELTAS[delta_index], size)


@dataclass
class Sample:
    corrupted: np.ndarray
    target: np.ndarray
    size: int
    tree_depth: int
    op: int
    address: int
    depth: int
    delta_index: int


def choose_program(rng: random.Random, split: str) -> Tuple[int, int, int, int]:
    if split in ("train", "iid"):
        tree_depth = rng.randint(4, 5)
        op_depth = rng.randint(1, 2)
        min_level, max_level = 0, min(2, tree_depth - op_depth)
    elif split == "ood_address":
        tree_depth = rng.randint(6, 8)
        op_depth = rng.randint(1, 2)
        min_level, max_level = 3, min(5, tree_depth - op_depth)
    elif split == "ood_depth":
        tree_depth = rng.randint(6, 8)
        op_depth = rng.randint(3, 4)
        min_level, max_level = 0, min(2, tree_depth - op_depth)
    elif split == "ood_joint":
        tree_depth = 8
        op_depth = rng.randint(3, 4)
        min_level, max_level = 3, tree_depth - op_depth
    else:
        raise ValueError(split)
    target_level = rng.randint(min_level, max_level)
    address = rng.randint(2 ** target_level, 2 ** (target_level + 1) - 1)
    return tree_depth, address, op_depth, rng.randrange(len(OPS))


def generate_samples(count: int, split: str, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        tree_depth, address, op_depth, op = choose_program(rng, split)
        root = rng.randint(1, 7)
        target = legal_heap(tree_depth, root)
        size = 2 ** (tree_depth + 1) - 1
        delta_index = rng.randrange(len(DELTAS))
        if op == 0:
            corrupted = mirror_subtree(target, address, op_depth, size)
            delta_index = 0
        else:
            corrupted = plus_subtree(target, address, op_depth, DELTAS[delta_index], size)
        samples.append(Sample(corrupted, target, size, tree_depth, op, address, op_depth, delta_index))
    return samples


class HeapDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            torch.from_numpy(sample.corrupted), torch.from_numpy(sample.target),
            sample.size, sample.op, sample.address - 1, sample.depth - 1,
            sample.delta_index,
        )


def algebra_features(arr: torch.Tensor, size: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = arr.shape[0]
    device = arr.device
    addresses = torch.arange(1, MAX_NODES + 1, device=device)
    valid = addresses[None] <= size[:, None]
    values = arr[:, 1:]
    parents = arr[:, (addresses // 2).clamp_min(1)]
    expected = 2 * parents + (addresses & 1).to(arr.dtype)[None]
    parent_residual = torch.where(addresses[None] == 1, torch.zeros_like(values), values - expected)

    left_index = addresses * 2
    right_index = left_index + 1
    left_valid = left_index[None] <= size[:, None]
    right_valid = right_index[None] <= size[:, None]
    left_value = arr[:, left_index.clamp_max(MAX_NODES)]
    right_value = arr[:, right_index.clamp_max(MAX_NODES)]
    left_residual = torch.where(left_valid, left_value - 2 * values, torch.zeros_like(values))
    right_residual = torch.where(right_valid, right_value - (2 * values + 1), torch.zeros_like(values))

    features = [
        values / 2048.0,
        parent_residual / 16.0,
        left_residual / 16.0,
        right_residual / 16.0,
        torch.full_like(values, 1.0),
    ]
    anomaly = parent_residual.abs() + left_residual.abs() + right_residual.abs()
    signed = parent_residual + left_residual + right_residual
    for relative_depth in range(MAX_OP_DEPTH + 1):
        abs_sum = torch.zeros_like(values)
        signed_sum = torch.zeros_like(values)
        count = torch.zeros_like(values)
        scale = 2 ** relative_depth
        for offset in range(scale):
            descendant = addresses * scale + offset
            descendant_valid = descendant[None] <= size[:, None]
            gather = (descendant - 1).clamp_max(MAX_NODES - 1)
            abs_sum += torch.where(descendant_valid, anomaly[:, gather], torch.zeros_like(values))
            signed_sum += torch.where(descendant_valid, signed[:, gather], torch.zeros_like(values))
            count += descendant_valid.to(values.dtype)
        features.extend([abs_sum / (16.0 * count.clamp_min(1)), signed_sum / (16.0 * count.clamp_min(1))])
    node_level = torch.floor(torch.log2(addresses.to(arr.dtype)))[None].expand(batch, -1) / MAX_TREE_DEPTH
    features.append(node_level)
    return torch.stack(features, dim=-1), valid


class StructuralProgramEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        feature_dim = 5 + 2 * (MAX_OP_DEPTH + 1) + 1
        self.kernel = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.address_head = nn.Linear(hidden, 1)
        self.op_head = nn.Linear(hidden, len(OPS))
        self.delta_head = nn.Linear(hidden, len(DELTAS))

    def forward(self, arr, size):
        features, valid = algebra_features(arr, size)
        hidden = self.kernel(features)
        address_logits = self.address_head(hidden).squeeze(-1).masked_fill(~valid, -1e9)
        # The corruption creates a residual wave whose maximum layer is the
        # operator depth.  Reading that layer is an algebraic recursive decoder,
        # so unseen depths do not require unseen categorical output weights.
        depth_logits = torch.stack([features[:, :, 5 + 2 * relative_depth] for relative_depth in range(1, MAX_OP_DEPTH + 1)], dim=-1) * 8.0
        return address_logits, self.op_head(hidden), depth_logits, self.delta_head(hidden)


class FlatProgramEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(MAX_NODES, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.address = nn.Linear(hidden, MAX_NODES)
        self.op = nn.Linear(hidden, len(OPS))
        self.depth = nn.Linear(hidden, MAX_OP_DEPTH)
        self.delta = nn.Linear(hidden, len(DELTAS))
    def forward(self, arr, size):
        scale = arr[:, 1:].abs().amax(-1, keepdim=True).clamp_min(1)
        hidden = self.body(arr[:, 1:] / scale)
        valid = torch.arange(MAX_NODES, device=arr.device)[None] < size[:, None]
        return self.address(hidden).masked_fill(~valid, -1e9), self.op(hidden), self.depth(hidden), self.delta(hidden)


class FlatRewriter(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(MAX_NODES, hidden), nn.GELU(), nn.Linear(hidden, MAX_NODES))
    def forward(self, arr):
        scale = arr[:, 1:].abs().amax(-1, keepdim=True).clamp_min(1)
        return self.net(arr[:, 1:] / scale) * scale


def program_loss(outputs, labels, structural: bool):
    address_logits, op_logits, depth_logits, delta_logits = outputs
    op, address, depth, delta = labels
    if structural:
        rows = torch.arange(address.shape[0], device=address.device)
        local_op = op_logits[rows, address]
        local_depth = depth_logits[rows, address]
        local_delta = delta_logits[rows, address]
    else:
        local_op, local_depth, local_delta = op_logits, depth_logits, delta_logits
    loss = F.cross_entropy(address_logits, address) + F.cross_entropy(local_op, op) + F.cross_entropy(local_depth, depth)
    plus_mask = op == 1
    if bool(plus_mask.any()):
        loss = loss + F.cross_entropy(local_delta[plus_mask], delta[plus_mask])
    return loss


def collapse_program(outputs, structural: bool):
    address_logits, op_logits, depth_logits, delta_logits = outputs
    address = address_logits.argmax(-1)
    if structural:
        rows = torch.arange(address.shape[0], device=address.device)
        op = op_logits[rows, address].argmax(-1)
        depth = depth_logits[rows, address].argmax(-1)
        delta = delta_logits[rows, address].argmax(-1)
    else:
        op, depth, delta = op_logits.argmax(-1), depth_logits.argmax(-1), delta_logits.argmax(-1)
    return op, address, depth, delta


@torch.no_grad()
def evaluate_program(model, samples, device, structural: bool):
    model.eval()
    loader = DataLoader(HeapDataset(samples), batch_size=128, shuffle=False)
    totals = {"samples": 0, "operator": 0, "address": 0, "depth": 0, "delta": 0, "program": 0, "restore": 0}
    if structural:
        totals.update({"operator_at_gold_address": 0, "depth_at_gold_address": 0, "delta_at_gold_address": 0})
    depth_confusion_at_gold = torch.zeros(MAX_OP_DEPTH, MAX_OP_DEPTH, dtype=torch.int64)
    examples = []
    for arr, target, size, op, address, depth, delta in loader:
        arr, size = arr.to(device), size.to(device)
        outputs = model(arr, size)
        pred_op, pred_address, pred_depth, pred_delta = collapse_program(outputs, structural)
        if structural:
            rows = torch.arange(arr.shape[0], device=device)
            gold_address = address.to(device)
            gold_local_op = outputs[1][rows, gold_address].argmax(-1).cpu()
            gold_local_depth = outputs[2][rows, gold_address].argmax(-1).cpu()
            gold_local_delta = outputs[3][rows, gold_address].argmax(-1).cpu()
            totals["operator_at_gold_address"] += int(gold_local_op.eq(op).sum())
            totals["depth_at_gold_address"] += int(gold_local_depth.eq(depth).sum())
            totals["delta_at_gold_address"] += int((gold_local_delta.eq(delta) | op.eq(0)).sum())
            for gold, predicted in zip(depth, gold_local_depth):
                depth_confusion_at_gold[int(gold), int(predicted)] += 1
        pred_address = pred_address + 1
        pred_depth = pred_depth + 1
        totals["samples"] += arr.shape[0]
        totals["operator"] += int(pred_op.cpu().eq(op).sum())
        totals["address"] += int(pred_address.cpu().eq(address + 1).sum())
        totals["depth"] += int(pred_depth.cpu().eq(depth + 1).sum())
        delta_ok = pred_delta.cpu().eq(delta) | op.eq(0)
        totals["delta"] += int(delta_ok.sum())
        program_ok = pred_op.cpu().eq(op) & pred_address.cpu().eq(address + 1) & pred_depth.cpu().eq(depth + 1) & delta_ok
        totals["program"] += int(program_ok.sum())
        for row in range(arr.shape[0]):
            restored = execute_inverse(arr[row].cpu().numpy(), int(pred_op[row]), int(pred_address[row]), int(pred_depth[row]), int(pred_delta[row]), int(size[row]))
            target_np = target[row].numpy()
            exact = np.array_equal(restored[:int(size[row]) + 1], target_np[:int(size[row]) + 1])
            totals["restore"] += int(exact)
            if len(examples) < 5:
                examples.append({
                    "gold": {"op": OPS[int(op[row])], "address": int(address[row]) + 1, "depth": int(depth[row]) + 1, "delta": DELTAS[int(delta[row])]},
                    "pred": {"op": OPS[int(pred_op[row])], "address": int(pred_address[row]), "depth": int(pred_depth[row]), "delta": DELTAS[int(pred_delta[row])]},
                    "restored_exact": exact,
                })
    n = totals.pop("samples")
    result = {**{key + "_accuracy": value / n for key, value in totals.items()}, "examples": examples}
    if structural:
        result["depth_confusion_at_gold_address"] = depth_confusion_at_gold.tolist()
    return result


@torch.no_grad()
def evaluate_rewriter(model, samples, device):
    model.eval()
    loader = DataLoader(HeapDataset(samples), batch_size=128, shuffle=False)
    full_exact = changed_exact = total = 0
    mse_sum = 0.0
    for arr, target, size, *_ in loader:
        arr, target, size = arr.to(device), target.to(device), size.to(device)
        prediction = model(arr)
        valid = torch.arange(MAX_NODES, device=device)[None] < size[:, None]
        target_values = target[:, 1:]
        mse_sum += float((((prediction - target_values) ** 2) * valid).sum().item())
        rounded = prediction.round()
        full_exact += int(((rounded.eq(target_values) | ~valid).all(-1)).sum().item())
        changed = arr[:, 1:].ne(target_values) & valid
        changed_exact += int(((rounded.eq(target_values) | ~changed).all(-1)).sum().item())
        total += arr.shape[0]
    return {"mse_per_valid_node": mse_sum / sum(s.size for s in samples), "full_restore_exact": full_exact / total, "changed_cells_exact": changed_exact / total}


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    train_samples = generate_samples(args.train_samples, "train", args.seed)
    splits = {
        "iid": generate_samples(args.test_samples, "iid", args.seed + 1),
        "ood_address": generate_samples(args.test_samples, "ood_address", args.seed + 2),
        "ood_depth": generate_samples(args.test_samples, "ood_depth", args.seed + 3),
        "ood_joint": generate_samples(args.test_samples, "ood_joint", args.seed + 4),
    }
    loader = DataLoader(HeapDataset(train_samples), batch_size=args.batch, shuffle=True)
    structural = StructuralProgramEncoder(args.hidden).to(device)
    flat = FlatProgramEncoder(args.hidden).to(device)
    rewriter = FlatRewriter(args.hidden).to(device)
    optimizers = {
        "structural": torch.optim.AdamW(structural.parameters(), lr=args.lr),
        "flat_program": torch.optim.AdamW(flat.parameters(), lr=args.lr),
        "flat_rewriter": torch.optim.AdamW(rewriter.parameters(), lr=args.lr),
    }
    trace = []
    for epoch in range(1, args.epochs + 1):
        sums = {key: 0.0 for key in optimizers}; batches = 0
        structural.train(); flat.train(); rewriter.train()
        for arr, target, size, op, address, depth, delta in loader:
            arr, target, size = arr.to(device), target.to(device), size.to(device)
            labels = (op.to(device), address.to(device), depth.to(device), delta.to(device))
            for name, model, is_structural in (("structural", structural, True), ("flat_program", flat, False)):
                loss = program_loss(model(arr, size), labels, is_structural)
                optimizers[name].zero_grad(set_to_none=True); loss.backward(); optimizers[name].step(); sums[name] += float(loss.item())
            prediction = rewriter(arr)
            valid = torch.arange(MAX_NODES, device=device)[None] < size[:, None]
            scale = target[:, 1:].abs().amax(-1, keepdim=True).clamp_min(1)
            rewrite_loss = ((((prediction - target[:, 1:]) / scale) ** 2) * valid).sum() / valid.sum()
            optimizers["flat_rewriter"].zero_grad(set_to_none=True); rewrite_loss.backward(); optimizers["flat_rewriter"].step(); sums["flat_rewriter"] += float(rewrite_loss.item())
            batches += 1
        row = {"epoch": epoch, **{name + "_loss": value / batches for name, value in sums.items()}}
        trace.append(row); print(json.dumps(row), flush=True)

    results = {}
    for split, samples in splits.items():
        results[split] = {
            "structural_program": evaluate_program(structural, samples, device, True),
            "flat_program": evaluate_program(flat, samples, device, False),
            "flat_rewriter": evaluate_rewriter(rewriter, samples, device),
            "oracle_executor_restore_exact": 1.0,
        }
    return structural, flat, rewriter, trace, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ara/m0-treeheap-math/evidence/algebraic_operator_codec_probe")
    ap.add_argument("--train-samples", type=int, default=8000)
    ap.add_argument("--test-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    structural, flat, rewriter, trace, results = train(args)
    summary = {
        "claim": "M0-OPCODEC-C01", "host": socket.gethostname(), "seconds": time.time() - started,
        "config": vars(args),
        "parameters": {"structural_program": sum(p.numel() for p in structural.parameters()), "flat_program": sum(p.numel() for p in flat.parameters()), "flat_rewriter": sum(p.numel() for p in rewriter.parameters())},
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace) + "\n", encoding="utf8")
    (out / "README.md").write_text("# Algebraic Operator Codec Probe\n\nSee `summary.json`.\n", encoding="utf8")
    torch.save({"state_dict": structural.state_dict(), "config": vars(args)}, out / "checkpoint_structural.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
