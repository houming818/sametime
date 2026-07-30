#!/usr/bin/env python3
"""Audit XOR-Butterfly communication over fixed-capacity TreeHeap addresses."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


CLAIM = "S3-TREEHEAP-BUTTERFLY-LONGRANGE-C01"


def power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def stages(width: int) -> int:
    if not power_of_two(width):
        raise ValueError(f"width must be a power of two, got {width}")
    return int(math.log2(width))


def pair_indices(width: int, stage: int, device: torch.device):
    bit = 1 << stage
    left = torch.arange(width, device=device)
    left = left[left.bitwise_and(bit).eq(0)]
    return left, left.bitwise_xor(bit)


def orthogonal_butterfly(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Butterfly over address bits, without an NxN matrix."""
    result = x
    scale = math.sqrt(2.0)
    for stage in range(stages(x.shape[1])):
        left_index, right_index = pair_indices(x.shape[1], stage, x.device)
        left, right = result[:, left_index], result[:, right_index]
        updated = result.clone()
        updated[:, left_index] = (left + right) / scale
        updated[:, right_index] = (left - right) / scale
        result = updated
    return result


def inverse_orthogonal_butterfly(x: torch.Tensor) -> torch.Tensor:
    result = x
    scale = math.sqrt(2.0)
    for stage in reversed(range(stages(x.shape[1]))):
        left_index, right_index = pair_indices(x.shape[1], stage, x.device)
        coarse, detail = result[:, left_index], result[:, right_index]
        updated = result.clone()
        updated[:, left_index] = (coarse + detail) / scale
        updated[:, right_index] = (coarse - detail) / scale
        result = updated
    return result


def influence_coverage(width: int) -> list[int]:
    influence = [{index} for index in range(width)]
    coverage = []
    for stage in range(stages(width)):
        bit = 1 << stage
        updated = [set(row) for row in influence]
        for left in range(width):
            if left & bit:
                continue
            right = left ^ bit
            merged = influence[left] | influence[right]
            updated[left] = set(merged)
            updated[right] = set(merged)
        influence = updated
        coverage.append(sum(len(row) for row in influence))
    return coverage


def deductive_contract(widths: Sequence[int], dim: int, device: torch.device):
    rows = []
    for width in widths:
        torch.manual_seed(9000 + width)
        source = torch.randn(4, width, dim, device=device, requires_grad=True)
        transformed = orthogonal_butterfly(source)
        restored = inverse_orthogonal_butterfly(transformed)
        inverse_mse = float((restored - source).square().mean().detach())
        source_energy = source.square().sum()
        transformed_energy = transformed.square().sum()
        energy_relative_error = float(
            ((transformed_energy - source_energy).abs() / source_energy).detach()
        )
        probe = torch.randn_like(transformed)
        gradient = torch.autograd.grad((transformed * probe).sum(), source)[0]
        gradient_norm_ratio = float(gradient.norm() / probe.norm())
        coverage = influence_coverage(width)
        rows.append({
            "width": width,
            "depth": stages(width),
            "inverse_mse": inverse_mse,
            "energy_relative_error": energy_relative_error,
            "gradient_norm_ratio": gradient_norm_ratio,
            "coverage_pairs_by_stage": coverage,
            "final_coverage_fraction": coverage[-1] / (width * width),
            "pair_operations": (width // 2) * stages(width),
            "dense_attention_pairs": width * width,
            "dense_attention_allocated": False,
            "largest_address_mixing_tensor_rank": 3,
        })
    return rows


def binary_features(values: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(bits, device=values.device)
    return values[:, None].bitwise_right_shift(shifts).bitwise_and(1).float()


def gather_xor(source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    address = torch.arange(source.shape[1], device=source.device)[None, :]
    index = address.bitwise_xor(query[:, None])
    return source.gather(1, index)


class TokenReadout(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        self.output.weight = self.embedding.weight

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(state)) / math.sqrt(state.shape[-1])


class ButterflyRouter(TokenReadout):
    """Shared query-bit value over changing XOR address partners."""

    def __init__(self, vocab: int, dim: int):
        super().__init__(vocab, dim)
        self.exchange_scale = nn.Parameter(torch.tensor(1.0))
        self.exchange_bias = nn.Parameter(torch.tensor(0.0))

    def exchange_probability(self, bit: torch.Tensor) -> torch.Tensor:
        signed = bit.mul(2.0).sub(1.0)
        return torch.sigmoid(self.exchange_scale * signed + self.exchange_bias)

    def forward(self, source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        state = self.embedding(source)
        query_bits = binary_features(query, stages(source.shape[1]))
        for stage in range(stages(source.shape[1])):
            left_index, right_index = pair_indices(source.shape[1], stage, source.device)
            left, right = state[:, left_index], state[:, right_index]
            exchange = self.exchange_probability(query_bits[:, stage])[:, None, None]
            updated = state.clone()
            updated[:, left_index] = (1.0 - exchange) * left + exchange * right
            updated[:, right_index] = exchange * left + (1.0 - exchange) * right
            state = updated
        return self.logits(state)

    def protocol(self):
        zero = torch.tensor(0.0, device=self.exchange_scale.device)
        one = torch.tensor(1.0, device=self.exchange_scale.device)
        return {
            "p_exchange_query_bit_0": float(self.exchange_probability(zero).detach()),
            "p_exchange_query_bit_1": float(self.exchange_probability(one).detach()),
        }


class AdjacentOnlyRouter(ButterflyRouter):
    """Same learned value, but every stage repeats only adjacent pairs."""

    def forward(self, source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        state = self.embedding(source)
        query_bits = binary_features(query, stages(source.shape[1]))
        left_index, right_index = pair_indices(source.shape[1], 0, source.device)
        for stage in range(stages(source.shape[1])):
            left, right = state[:, left_index], state[:, right_index]
            exchange = self.exchange_probability(query_bits[:, stage])[:, None, None]
            updated = state.clone()
            updated[:, left_index] = (1.0 - exchange) * left + exchange * right
            updated[:, right_index] = exchange * left + (1.0 - exchange) * right
            state = updated
        return self.logits(state)


class RootBottleneckRouter(TokenReadout):
    """Finite root summary followed by address-conditioned parallel readout."""

    def __init__(self, vocab: int, dim: int, max_bits: int):
        super().__init__(vocab, dim)
        self.fold = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim),
        )
        self.address = nn.Linear(max_bits, dim, bias=False)
        self.query = nn.Linear(max_bits, dim, bias=False)
        self.refine = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.max_bits = max_bits

    def forward(self, source: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        state = self.embedding(source)
        width = source.shape[1]
        while state.shape[1] > 1:
            state = self.fold(torch.cat((state[:, 0::2], state[:, 1::2]), dim=-1))
        root = state[:, 0]
        position = torch.arange(width, device=source.device)
        address_bits = binary_features(position, self.max_bits)
        query_bits = binary_features(query, self.max_bits)
        leaf = (
            root[:, None]
            + self.address(address_bits)[None]
            + self.query(query_bits)[:, None]
        )
        leaf = leaf + self.refine(leaf)
        return self.logits(leaf)


def query_sets(width: int):
    masks = list(range(1, width))
    train = [value for value in masks if value.bit_count() <= 2]
    heldout = [value for value in masks if value.bit_count() >= 3]
    return train, heldout


def sample_batch(
    batch: int, width: int, vocab: int, queries: Sequence[int], device: torch.device,
):
    source = torch.randint(vocab, (batch, width), device=device)
    query_table = torch.tensor(queries, device=device)
    query = query_table[torch.randint(len(queries), (batch,), device=device)]
    target = gather_xor(source, query)
    return source, query, target


@torch.no_grad()
def evaluate(
    model: nn.Module, width: int, vocab: int, queries: Sequence[int],
    batch: int, batches: int, device: torch.device,
):
    model.eval()
    loss_sum = correct = tokens = 0
    by_weight: dict[int, list[int]] = {}
    for _ in range(batches):
        source, query, target = sample_batch(batch, width, vocab, queries, device)
        logits = model(source, query)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, vocab), target.reshape(-1), reduction="sum",
        ))
        predicted = logits.argmax(-1)
        match = predicted.eq(target)
        correct += int(match.sum())
        tokens += target.numel()
        for weight in sorted({int(value.bit_count()) for value in queries}):
            selected = torch.tensor(
                [int(value.bit_count()) == weight for value in query.tolist()],
                device=device,
            )
            if not bool(selected.any()):
                continue
            row = by_weight.setdefault(weight, [0, 0])
            row[0] += int(match[selected].sum())
            row[1] += int(match[selected].numel())
    return {
        "nll": loss_sum / max(1, tokens),
        "token_accuracy": correct / max(1, tokens),
        "tokens": tokens,
        "accuracy_by_query_hamming_weight": {
            str(key): value[0] / max(1, value[1]) for key, value in by_weight.items()
        },
    }


def model_for(name: str, vocab: int, dim: int, max_bits: int):
    if name == "butterfly":
        return ButterflyRouter(vocab, dim)
    if name == "adjacent_only":
        return AdjacentOnlyRouter(vocab, dim)
    if name == "root_bottleneck":
        return RootBottleneckRouter(vocab, dim, max_bits)
    raise ValueError(name)


def train_arm(name: str, seed: int, args, output: Path, trace_path: Path):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = model_for(name, args.vocab, args.dim, stages(args.ood_width)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_queries, heldout_queries = query_sets(args.train_width)
    _, ood_queries = query_sets(args.ood_width)
    checkpoints = {1, args.steps // 4, args.steps // 2, args.steps}
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        source, query, target = sample_batch(
            args.batch, args.train_width, args.vocab, train_queries, device,
        )
        logits = model(source, query)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step not in checkpoints:
            continue
        row = {
            "event": "train", "arm": name, "seed": seed, "step": step,
            "train_nll": float(loss.detach()), "elapsed_sec": time.time() - started,
        }
        if hasattr(model, "protocol"):
            row["protocol"] = model.protocol()
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    result = {
        "arm": name,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.time() - started,
        "train_queries": train_queries,
        "heldout_queries": heldout_queries,
        "train_width": evaluate(
            model, args.train_width, args.vocab, train_queries,
            args.eval_batch, args.eval_batches, device,
        ),
        "heldout_width": evaluate(
            model, args.train_width, args.vocab, heldout_queries,
            args.eval_batch, args.eval_batches, device,
        ),
        "max_distance": evaluate(
            model, args.train_width, args.vocab, [args.train_width - 1],
            args.eval_batch, args.eval_batches, device,
        ),
        "ood_width": evaluate(
            model, args.ood_width, args.vocab, ood_queries,
            args.eval_batch, args.eval_batches, device,
        ),
        "ood_max_distance": evaluate(
            model, args.ood_width, args.vocab, [args.ood_width - 1],
            args.eval_batch, args.eval_batches, device,
        ),
    }
    if hasattr(model, "protocol"):
        result["protocol"] = model.protocol()
        result["query_bit_probability_gap"] = (
            result["protocol"]["p_exchange_query_bit_1"]
            - result["protocol"]["p_exchange_query_bit_0"]
        )
    checkpoint = output / f"{name}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "result": result}, checkpoint)
    result["checkpoint"] = str(checkpoint)
    return result


def mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return sum(values) / max(1, len(values))


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_treeheap_butterfly_long_range")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="8101,8102,8103")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--train-width", type=int, default=32)
    parser.add_argument("--ood-width", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = args.seeds.split(",")[0]
        args.steps = min(args.steps, 120)
        args.batch = min(args.batch, 64)
        args.eval_batch = min(args.eval_batch, 64)
        args.eval_batches = min(args.eval_batches, 3)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    device = torch.device(args.device)
    algebra = deductive_contract((8, 16, 32, 64), args.dim, device)
    rows = []
    for seed in seeds:
        for arm in ("butterfly", "adjacent_only", "root_bottleneck"):
            rows.append(train_arm(arm, seed, args, output, trace_path))

    grouped = {
        arm: [row for row in rows if row["arm"] == arm]
        for arm in ("butterfly", "adjacent_only", "root_bottleneck")
    }
    aggregate = {}
    for arm, arm_rows in grouped.items():
        aggregate[arm] = {
            "mean_train_accuracy": mean(row["train_width"]["token_accuracy"] for row in arm_rows),
            "mean_heldout_accuracy": mean(row["heldout_width"]["token_accuracy"] for row in arm_rows),
            "mean_max_distance_accuracy": mean(row["max_distance"]["token_accuracy"] for row in arm_rows),
            "mean_ood_accuracy": mean(row["ood_width"]["token_accuracy"] for row in arm_rows),
            "mean_ood_max_distance_accuracy": mean(row["ood_max_distance"]["token_accuracy"] for row in arm_rows),
            "mean_parameters": mean(row["parameters"] for row in arm_rows),
            "mean_seconds": mean(row["seconds"] for row in arm_rows),
        }
    butterfly_rows = grouped["butterfly"]
    algebra_pass = all(
        row["inverse_mse"] <= 1e-10
        and row["energy_relative_error"] <= 1e-6
        and 0.999 <= row["gradient_norm_ratio"] <= 1.001
        and row["final_coverage_fraction"] == 1.0
        for row in algebra
    )
    per_seed_pass = []
    for row in butterfly_rows:
        adjacent = next(
            item for item in grouped["adjacent_only"] if item["seed"] == row["seed"]
        )
        root = next(
            item for item in grouped["root_bottleneck"] if item["seed"] == row["seed"]
        )
        per_seed_pass.append({
            "seed": row["seed"],
            "heldout_at_least_0_95": row["heldout_width"]["token_accuracy"] >= 0.95,
            "max_distance_at_least_0_90": row["max_distance"]["token_accuracy"] >= 0.90,
            "ood_at_least_0_90": row["ood_width"]["token_accuracy"] >= 0.90,
            "beats_adjacent_by_0_25": (
                row["heldout_width"]["token_accuracy"]
                - adjacent["heldout_width"]["token_accuracy"] >= 0.25
            ),
            "beats_root_by_0_25": (
                row["heldout_width"]["token_accuracy"]
                - root["heldout_width"]["token_accuracy"] >= 0.25
            ),
            "query_bit_probability_gap_at_least_0_50": (
                row.get("query_bit_probability_gap", 0.0) >= 0.50
            ),
        })
    seed_full_pass = [all(value for key, value in row.items() if key != "seed") for row in per_seed_pass]
    full_pass = algebra_pass and sum(seed_full_pass) >= 2
    summary = {
        "claim": CLAIM,
        "status": "supported_mechanism" if full_pass else (
            "algebra_only" if algebra_pass else "rejected"
        ),
        "scope": "synthetic address-conditioned long-range communication only",
        "config": vars(args),
        "environment": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_revision_before_run": git_revision(),
            "argv": [shlex.quote(value) for value in os.sys.argv],
        },
        "deductive_contract": algebra,
        "runs": rows,
        "aggregate": aggregate,
        "gates": {
            "deductive_contract": algebra_pass,
            "per_seed": per_seed_pass,
            "seed_full_pass": seed_full_pass,
            "at_least_two_of_three_seeds": sum(seed_full_pass) >= 2,
            "full_claim": full_pass,
        },
        "not_proved": [
            "language semantics",
            "WMT or dialogue improvement",
            "compression or wall-clock superiority",
            "superiority over a tuned Transformer",
            "emergent private protocol",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    command = "python3 " + " ".join(shlex.quote(value) for value in os.sys.argv)
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Butterfly long-range evidence\n\n"
        f"Claim: `{CLAIM}`\n\n"
        f"Status: `{summary['status']}`\n\n"
        "This directory records an algebra contract and a synthetic learned "
        "address-routing probe. It is not language evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "claim": CLAIM,
        "status": summary["status"],
        "aggregate": aggregate,
        "gates": summary["gates"],
        "output": str(output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
