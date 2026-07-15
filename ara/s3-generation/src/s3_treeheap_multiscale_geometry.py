#!/usr/bin/env python3
"""Proof probe for scale-indexed geometry in a recursive TreeHeap.

The model is never asked to reconstruct leaf tokens from an upper node.  A
shared recursive FOLD kernel instead learns two exact, label-free observables
of every receptive field: an unordered token sketch and an ordered adjacency
sketch.  Held-out nearest-neighbour tests ask whether the learned state itself
forms a useful geometric index, rather than merely supporting a readout head.
"""

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
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("treeheap_geometry_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import data helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiscaleTreeHeap(nn.Module):
    """One WRITE table, one address-shared FOLD kernel, and two READ kernels."""

    def __init__(self, vocab: int, dim: int, sketch_dim: int, max_depth: int):
        super().__init__()
        self.write = nn.Embedding(vocab, dim)
        self.depth = nn.Embedding(max_depth + 1, dim)
        self.fold = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.bag_read = nn.Linear(dim, sketch_dim)
        self.adj_read = nn.Linear(dim, sketch_dim)

    def levels(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        node = self.write(tokens)
        out: List[torch.Tensor] = []
        depth = 1
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            d = self.depth.weight[depth].view(1, 1, -1).expand_as(left)
            node = self.norm(self.fold(torch.cat((left, right, d), dim=-1)))
            out.append(node)
            depth += 1
        return out


class ExactSketches:
    """Fixed corpus observables with exact recursive compose laws."""

    def __init__(self, vocab: int, dim: int, seed: int, device: torch.device):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scale = dim ** -0.5
        self.bag = torch.randn(vocab, dim, generator=generator) * scale
        self.pair_left = torch.randn(vocab, dim, generator=generator) * scale
        self.pair_right = torch.randn(vocab, dim, generator=generator) * scale
        self.bag = self.bag.to(device)
        self.pair_left = self.pair_left.to(device)
        self.pair_right = self.pair_right.to(device)
        self.pair_scale = math.sqrt(dim)

    def pair(self, left_token: torch.Tensor, right_token: torch.Tensor) -> torch.Tensor:
        return self.pair_left[left_token] * self.pair_right[right_token] * self.pair_scale

    def levels(self, tokens: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        bag = self.bag[tokens]
        adj = torch.zeros_like(bag)
        first = tokens
        last = tokens
        bag_levels, adj_levels = [], []
        while bag.shape[1] > 1:
            left_bag, right_bag = bag[:, 0::2], bag[:, 1::2]
            left_adj, right_adj = adj[:, 0::2], adj[:, 1::2]
            left_first, right_first = first[:, 0::2], first[:, 1::2]
            left_last, right_last = last[:, 0::2], last[:, 1::2]
            bag = left_bag + right_bag
            adj = left_adj + right_adj + self.pair(left_last, right_first)
            first, last = left_first, right_last
            bag_levels.append(bag)
            adj_levels.append(adj)
        return bag_levels, adj_levels

    def direct(self, spans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bag = self.bag[spans].sum(dim=-2)
        adj = self.pair(spans[..., :-1], spans[..., 1:]).sum(dim=-2)
        return bag, adj


def batches(base, block_dir: Path, split: str, batch: int, seed: int, limit: int):
    yield from base.iter_batches(block_dir, base.manifest(block_dir, split), batch, seed, limit)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()


def algebra_audit(sketch: ExactSketches, tokens: torch.Tensor) -> dict:
    recursive_bag, recursive_adj = sketch.levels(tokens)
    bag_error = adj_error = 0.0
    for depth, (bag, adj) in enumerate(zip(recursive_bag, recursive_adj), start=1):
        width = 2 ** depth
        spans = tokens.reshape(tokens.shape[0], -1, width)
        direct_bag, direct_adj = sketch.direct(spans)
        bag_error = max(bag_error, float((bag.double() - direct_bag.double()).abs().max()))
        adj_error = max(adj_error, float((adj.double() - direct_adj.double()).abs().max()))
    return {"bag_max_abs_error": bag_error, "adjacency_max_abs_error": adj_error}


def train(model, sketch, base, args, device: torch.device) -> List[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    started = time.time()
    for step, (tokens, _) in enumerate(
        batches(base, Path(args.block_dir), "train", args.batch, args.seed + 1, args.train_blocks), start=1
    ):
        tokens = tokens.to(device)
        states = model.levels(tokens)
        bag, adj = sketch.levels(tokens)
        task_index = step - 1
        depth_index = (task_index // 2) % args.depths
        is_bag = task_index % 2 == 0
        state = states[depth_index]
        target = bag[depth_index] if is_bag else adj[depth_index]
        pred = model.bag_read(state) if is_bag else model.adj_read(state)
        loss = cosine_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "task": "bag" if is_bag else "adjacency",
                "depth": depth_index + 1,
                "loss": float(loss.detach()),
                "elapsed_sec": time.time() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return history


def capped_rows(tensor: torch.Tensor, cap: int) -> torch.Tensor:
    if tensor.ndim > 2:
        tensor = tensor.flatten(0, -2)
    tensor = tensor.detach().cpu()
    if tensor.shape[0] <= cap:
        return tensor
    index = torch.linspace(0, tensor.shape[0] - 1, cap).long()
    return tensor[index]


def neighbour_gain(state: torch.Tensor, target: torch.Tensor) -> dict:
    state = F.normalize(state.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    count = state.shape[0]
    similarity = state @ state.T
    similarity.fill_diagonal_(-float("inf"))
    nearest = similarity.argmax(dim=-1)
    random_index = (torch.arange(count) + max(1, count // 3)) % count
    nn_cos = (target * target[nearest]).sum(-1)
    random_cos = (target * target[random_index]).sum(-1)
    return {
        "nearest_target_cosine": float(nn_cos.mean()),
        "random_target_cosine": float(random_cos.mean()),
        "gain": float((nn_cos - random_cos).mean()),
    }


@torch.no_grad()
def evaluate(model, sketch, base, args, device: torch.device) -> dict:
    model.eval()
    state_rows: Dict[int, List[torch.Tensor]] = {d: [] for d in range(1, args.depths + 1)}
    bag_rows: Dict[int, List[torch.Tensor]] = {d: [] for d in range(1, args.depths + 1)}
    adj_rows: Dict[int, List[torch.Tensor]] = {d: [] for d in range(1, args.depths + 1)}
    bag_cos = {d: [] for d in range(1, args.depths + 1)}
    adj_cos = {d: [] for d in range(1, args.depths + 1)}
    root_bag_invariance, root_adj_invariance = [], []
    first_batch = None

    for tokens, _ in batches(base, Path(args.block_dir), "valid", args.eval_batch, args.seed + 2, args.valid_blocks):
        tokens = tokens.to(device)
        if first_batch is None:
            first_batch = tokens[: min(8, tokens.shape[0])]
        states = model.levels(tokens)
        bags, adjs = sketch.levels(tokens)
        for depth_index, (state, bag, adj) in enumerate(zip(states, bags, adjs), start=1):
            bag_pred = model.bag_read(state)
            adj_pred = model.adj_read(state)
            bag_cos[depth_index].append(F.cosine_similarity(bag_pred, bag, dim=-1).flatten().cpu())
            adj_cos[depth_index].append(F.cosine_similarity(adj_pred, adj, dim=-1).flatten().cpu())
            state_rows[depth_index].append(state.detach().flatten(0, 1).cpu())
            bag_rows[depth_index].append(bag.detach().flatten(0, 1).cpu())
            adj_rows[depth_index].append(adj.detach().flatten(0, 1).cpu())

        shuffled = tokens.flip(dims=(1,))
        shuffled_root = model.levels(shuffled)[-1]
        root = states[-1]
        root_bag_invariance.append(F.cosine_similarity(model.bag_read(root), model.bag_read(shuffled_root), dim=-1).cpu())
        root_adj_invariance.append(F.cosine_similarity(model.adj_read(root), model.adj_read(shuffled_root), dim=-1).cpu())

    metrics = {}
    for depth in range(1, args.depths + 1):
        states = capped_rows(torch.cat(state_rows[depth]), args.query_samples)
        bags = capped_rows(torch.cat(bag_rows[depth]), args.query_samples)
        adjs = capped_rows(torch.cat(adj_rows[depth]), args.query_samples)
        metrics[str(depth)] = {
            "receptive_tokens": 2 ** depth,
            "bag_read_cosine": float(torch.cat(bag_cos[depth]).mean()),
            "adjacency_read_cosine": float(torch.cat(adj_cos[depth]).mean()),
            "bag_query": neighbour_gain(states, bags),
            "adjacency_query": neighbour_gain(states, adjs),
            "query_samples": int(states.shape[0]),
        }

    bag_inv = float(torch.cat(root_bag_invariance).mean())
    adj_inv = float(torch.cat(root_adj_invariance).mean())
    return {
        "depths": metrics,
        "root_shuffle": {
            "bag_prediction_invariance": bag_inv,
            "adjacency_prediction_invariance": adj_inv,
            "invariance_gap": bag_inv - adj_inv,
        },
        "algebra": algebra_audit(sketch, first_batch),
    }


def gradient_audit(model, sketch, tokens: torch.Tensor) -> dict:
    model.train()
    result = {}
    for task in ("bag", "adjacency"):
        model.zero_grad(set_to_none=True)
        states = model.levels(tokens)
        bags, adjs = sketch.levels(tokens)
        pred = model.bag_read(states[-1]) if task == "bag" else model.adj_read(states[-1])
        target = bags[-1] if task == "bag" else adjs[-1]
        cosine_loss(pred, target).backward()
        result[task] = {
            "write": float(model.write.weight.grad.norm()),
            "fold": float(sum(p.grad.norm() for p in model.fold.parameters() if p.grad is not None)),
            "depth": float(model.depth.weight.grad.norm()),
            "read": float((model.bag_read if task == "bag" else model.adj_read).weight.grad.norm()),
        }
    return result


def decide(metrics: dict, gradients: dict) -> dict:
    depths = metrics["depths"]
    p1 = max(metrics["algebra"].values()) <= 1e-6
    p2 = all(row["bag_read_cosine"] >= 0.80 and row["adjacency_read_cosine"] >= 0.60 for row in depths.values())
    p3 = all(depths[str(d)]["bag_query"]["gain"] >= 0.10 and depths[str(d)]["adjacency_query"]["gain"] >= 0.05 for d in (4, 5, 6))
    p4 = metrics["root_shuffle"]["invariance_gap"] >= 0.10
    p5 = all(row["bag_query"]["gain"] > 0 and row["adjacency_query"]["gain"] > 0 for row in depths.values())
    grad_values = [value for task in gradients.values() for value in task.values()]
    p6 = all(math.isfinite(value) and value > 0 for value in grad_values)
    gates = {f"P{i}": value for i, value in enumerate((p1, p2, p3, p4, p5, p6), start=1)}
    if all(gates.values()):
        status = "supported_single_seed"
    elif p1 and p2 and not (p3 and p5):
        status = "partial_readable_not_indexed"
    else:
        status = "not_supported_by_this_probe"
    return {"gates": gates, "passed": sum(gates.values()), "total": len(gates), "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", default=str(Path(__file__).with_name("s3_residual_treeheap_forest_pretrain.py")))
    parser.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_treeheap_multiscale_geometry_smoke")
    parser.add_argument("--seed", type=int, default=71501)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--train-blocks", type=int, default=200000)
    parser.add_argument("--valid-blocks", type=int, default=512)
    parser.add_argument("--query-samples", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    base = load_base(Path(args.base_script))
    train_manifest = base.manifest(Path(args.block_dir), "train")
    vocab = int(train_manifest["tokenizer"]["vocab"])
    model = MultiscaleTreeHeap(vocab, args.dim, args.sketch_dim, args.depths).to(device)
    sketch = ExactSketches(vocab, args.sketch_dim, args.seed + 100, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    history = train(model, sketch, base, args, device)
    metrics = evaluate(model, sketch, base, args, device)
    first_tokens, _ = next(batches(base, Path(args.block_dir), "valid", min(16, args.eval_batch), args.seed + 3, 16))
    gradients = gradient_audit(model, sketch, first_tokens.to(device))
    decision = decide(metrics, gradients)
    summary = {
        "claim": "S3-TREEHEAP-GEOMETRY-C01",
        "predict": "P-S3-TREEHEAP-GEOMETRY-01",
        "host": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "config": vars(args),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
        "gradients": gradients,
        "decision": decision,
        "boundary": "Bag and directed-adjacency multiscale geometry only; not token reconstruction, semantics, or world knowledge.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "trace.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in history) + "\n", encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": vars(args)}, output / "checkpoint.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
