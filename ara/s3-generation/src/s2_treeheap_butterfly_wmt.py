#!/usr/bin/env python3
"""Matched WMT ablation for reversible TreeHeap Butterfly communication."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as data_base
import s2_lifting_pump_wmt as lifting
import s2_adaptive_lifting_wmt as adaptive


CLAIM = "S3-TREEHEAP-BUTTERFLY-WMT-C02"
ARMS = ("identity", "adjacent", "butterfly")


def pair_indices(width: int, stage: int, device: torch.device):
    bit = 1 << stage
    left = torch.arange(width, device=device)
    left = left[left.bitwise_and(bit).eq(0)]
    return left, left.bitwise_xor(bit)


class CouplingFunction(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(value))


class ReversibleAddressCommunication(nn.Module):
    """Shared additive coupling over either repeated or XOR address pairs."""

    def __init__(self, dim: int, depths: int, scale: float):
        super().__init__()
        self.forward_kernel = CouplingFunction(dim)
        self.backward_kernel = CouplingFunction(dim)
        self.depth_gain = nn.Parameter(torch.ones(depths))
        self.depths = depths
        self.scale = scale

    def schedule(self, mode: str, active_depths: int | None = None) -> List[int]:
        depths = self.depths if active_depths is None else active_depths
        if mode == "identity":
            return []
        if mode == "adjacent":
            return [0] * depths
        if mode == "butterfly":
            return list(range(depths))
        raise ValueError(mode)

    def _gain(self, step: int) -> torch.Tensor:
        return self.scale * torch.tanh(self.depth_gain[step])

    def forward(self, state: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
        result = state
        active_depths = int(math.log2(result.shape[1]))
        for step, address_stage in enumerate(self.schedule(mode, active_depths)):
            left_index, right_index = pair_indices(
                result.shape[1], address_stage, result.device,
            )
            left, right = result[:, left_index], result[:, right_index]
            active = (mask[:, left_index] & mask[:, right_index])[:, :, None]
            gain = self._gain(step)
            right_next = right + gain * self.forward_kernel(left)
            left_next = left + gain * self.backward_kernel(right_next)
            updated = result.clone()
            updated[:, left_index] = torch.where(active, left_next, left)
            updated[:, right_index] = torch.where(active, right_next, right)
            result = updated
        return result

    def inverse(self, state: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
        result = state
        active_depths = int(math.log2(result.shape[1]))
        schedule = self.schedule(mode, active_depths)
        for step in reversed(range(len(schedule))):
            address_stage = schedule[step]
            left_index, right_index = pair_indices(
                result.shape[1], address_stage, result.device,
            )
            left_next, right_next = result[:, left_index], result[:, right_index]
            active = (mask[:, left_index] & mask[:, right_index])[:, :, None]
            gain = self._gain(step)
            left = left_next - gain * self.backward_kernel(right_next)
            right = right_next - gain * self.forward_kernel(left)
            updated = result.clone()
            updated[:, left_index] = torch.where(active, left, left_next)
            updated[:, right_index] = torch.where(active, right, right_next)
            result = updated
        return result


class ButterflyAdaptiveEncoder(adaptive.AdaptiveLiftingEncoder):
    def __init__(
        self, vocab: int, dim: int, heap_width: int, pad: int, mode: str,
        scale: float, dynamic_width: bool = False,
    ):
        super().__init__(vocab, dim, heap_width, pad, learned_update=True, alternate=False)
        self.communication = ReversibleAddressCommunication(dim, self.depths, scale)
        self.communication_mode = mode
        self.runtime_mode: str | None = None
        self.dynamic_width = dynamic_width

    def raw_leaf(self, src: torch.Tensor, length: torch.Tensor):
        width = self.heap_width
        if self.dynamic_width:
            required = max(2, int(length.max().item()))
            width = 1 << (required - 1).bit_length()
            width = min(width, self.heap_width)
        padded = torch.full(
            (src.shape[0], width), self.pad,
            dtype=src.dtype, device=src.device,
        )
        padded[:, : src.shape[1]] = src
        mask = torch.arange(width, device=src.device)[None] < length[:, None]
        return self.embedding(padded) * mask[:, :, None], mask

    def fold(self, src: torch.Tensor, length: torch.Tensor, pair_break_depth: int = -1):
        if src.shape[1] > self.heap_width:
            raise ValueError(f"source width {src.shape[1]} exceeds heap width {self.heap_width}")
        leaf, leaf_mask = self.raw_leaf(src, length)
        mode = self.runtime_mode or self.communication_mode
        leaf = self.communication(leaf, leaf_mask, mode)
        node, node_mask = leaf, leaf_mask
        details: List[torch.Tensor] = []
        masks: List[torch.Tensor] = [leaf_mask]
        active_depths = int(math.log2(leaf.shape[1]))
        for depth in range(active_depths):
            left, right = node[:, 0::2], node[:, 1::2]
            lm, rm = node_mask[:, 0::2], node_mask[:, 1::2]
            if depth == pair_break_depth:
                right, rm = right.roll(1, dims=0), rm.roll(1, dims=0)
            detail = right - self.predictor(left)
            parent = left + self.update(detail)
            node_mask = lm | rm
            parent = parent * node_mask[:, :, None]
            detail = detail * node_mask[:, :, None]
            details.append(detail)
            masks.append(node_mask)
            node = parent
        return leaf, node[:, 0], details, masks


class ButterflyRecursive(adaptive.AdaptiveRecursive):
    def __init__(
        self, vocab: int, dim: int, hidden: int, heap_width: int, pad: int,
        mode: str, scale: float, dynamic_width: bool = False,
    ):
        nn.Module.__init__(self)
        self.encoder = ButterflyAdaptiveEncoder(
            vocab, dim, heap_width, pad, mode, scale, dynamic_width,
        )
        self.decoder = lifting.RecursiveDecoder(vocab, dim, hidden, self.encoder.depths)


def make_loaders(rows, args, pad: int, seed: int):
    splits = [
        rows[: args.train_samples],
        rows[args.train_samples : args.train_samples + args.valid_samples],
        rows[args.train_samples + args.valid_samples :],
    ]
    loaders = []
    for index, split in enumerate(splits):
        generator = torch.Generator().manual_seed(seed + 991) if index == 0 else None
        loaders.append(DataLoader(
            data_base.ParallelDataset(split),
            batch_size=args.batch_size,
            shuffle=index == 0,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=data_base.collate(pad),
            pin_memory=args.device.startswith("cuda"),
        ))
    long_rows = [row for row in splits[2] if len(row[0]) - 1 >= args.long_min]
    long_loader = DataLoader(
        data_base.ParallelDataset(long_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_base.collate(pad),
        pin_memory=args.device.startswith("cuda"),
    )
    return loaders, long_loader, len(long_rows)


@torch.no_grad()
def communication_audit(model: ButterflyRecursive, loader, device: str):
    src, length, _, _ = next(iter(loader))
    src, length = src.to(device), length.to(device)
    raw, mask = model.encoder.raw_leaf(src, length)
    native = model.encoder.communication_mode
    transformed = model.encoder.communication(raw, mask, native)
    restored = model.encoder.communication.inverse(transformed, mask, native)
    return {
        "initial_or_trained_delta_rms": float((transformed - raw).square().mean().sqrt()),
        "inverse_mse": float((restored - raw).square().mean()),
        "inverse_max_abs": float((restored - raw).abs().max()),
        "dense_attention_allocated": False,
        "pair_operations_per_example": (
            0 if native == "identity"
            else (model.encoder.heap_width // 2) * model.encoder.depths
        ),
    }


@torch.no_grad()
def evaluate_override(model, loader, args, pad, bos, eos, sp, mode: str):
    previous = model.encoder.runtime_mode
    model.encoder.runtime_mode = mode
    try:
        return lifting.evaluate(model, loader, args, pad, bos, eos, sp)
    finally:
        model.encoder.runtime_mode = previous


def train_arm(
    arm: str, seed: int, rows, args, vocab: int, pad: int, bos: int, eos: int,
    sp, output: Path,
):
    torch.manual_seed(seed + 17011)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 17011)
    model = ButterflyRecursive(
        vocab, args.dim, args.hidden, args.heap_width, pad, arm, args.coupling_scale,
    ).to(args.device)
    loaders, long_loader, long_count = make_loaders(rows, args, pad, seed)
    train_loader, valid_loader, test_loader = loaders
    initial_comm = communication_audit(model, test_loader, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trace = []
    best_nll, best = float("inf"), None
    gradients_ok = True
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = steps = 0
        for src, length, target, _ in train_loader:
            src = src.to(args.device, non_blocking=True)
            length = length.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            logits, _ = model.teacher(src, length, target, bos)
            loss = data_base.ce(logits, target, pad)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_ok = gradients_ok and lifting.finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        valid = lifting.evaluate(model, valid_loader, args, pad, bos, eos, sp)
        row = {
            "seed": seed,
            "arm": arm,
            "epoch": epoch,
            "train_nll": loss_sum / max(1, steps),
            "valid_nll": valid["nll"],
            "elapsed_sec": time.time() - started,
        }
        trace.append(row)
        print(json.dumps(row), flush=True)
        if valid["nll"] < best_nll:
            best_nll = valid["nll"]
            best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is None:
        raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "finite_gradients": gradients_ok,
        "seconds": time.time() - started,
        "initial_communication": initial_comm,
        "trained_communication": communication_audit(model, test_loader, args.device),
        "closure": adaptive.closure_audit(model, test_loader, args.device),
        "update": adaptive.update_audit(model, test_loader, args.device),
        "trace": trace,
        "test": lifting.evaluate(
            model, test_loader, args, pad, bos, eos, sp,
            generate=args.generate_examples,
        ),
        "long_test": lifting.evaluate(model, long_loader, args, pad, bos, eos, sp),
        "long_test_rows": long_count,
    }
    if arm == "butterfly":
        result["interventions"] = {
            "communication_identity": evaluate_override(
                model, test_loader, args, pad, bos, eos, sp, "identity",
            ),
            "communication_adjacent": evaluate_override(
                model, test_loader, args, pad, bos, eos, sp, "adjacent",
            ),
            "source_shuffle": lifting.evaluate(
                model, test_loader, args, pad, bos, eos, sp,
                intervention="source_shuffle",
            ),
        }
        if args.save_checkpoint:
            checkpoint = output / f"checkpoint_seed{seed}_butterfly.pt"
            torch.save({
                "claim": CLAIM,
                "seed": seed,
                "arm": arm,
                "state_dict": best,
                "config": vars(args),
            }, checkpoint)
            result["checkpoint"] = checkpoint.name
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def seed_decision(result: Dict[str, dict]):
    identity = result["identity"]
    adjacent = result["adjacent"]
    butterfly = result["butterfly"]
    butterfly_nll = butterfly["test"]["nll"]
    all_gain = identity["test"]["nll"] - butterfly_nll
    long_gain = identity["long_test"]["nll"] - butterfly["long_test"]["nll"]
    route = butterfly["test"].get("route_depth_mass", [])
    derived = {
        "identity_nll": identity["test"]["nll"],
        "adjacent_nll": adjacent["test"]["nll"],
        "butterfly_nll": butterfly_nll,
        "butterfly_gain_over_identity": all_gain,
        "butterfly_gain_over_adjacent": adjacent["test"]["nll"] - butterfly_nll,
        "identity_long_nll": identity["long_test"]["nll"],
        "butterfly_long_nll": butterfly["long_test"]["nll"],
        "butterfly_long_gain": long_gain,
        "disable_damage": (
            butterfly["interventions"]["communication_identity"]["nll"] - butterfly_nll
        ),
        "adjacent_override_damage": (
            butterfly["interventions"]["communication_adjacent"]["nll"] - butterfly_nll
        ),
        "source_shuffle_damage": (
            butterfly["interventions"]["source_shuffle"]["nll"] - butterfly_nll
        ),
    }
    gates = {
        "P1_butterfly_beats_identity": derived["butterfly_gain_over_identity"] >= 0.02,
        "P2_butterfly_beats_adjacent": derived["butterfly_gain_over_adjacent"] >= 0.015,
        "P3_long_source_gain": (
            long_gain >= 0.02 and long_gain >= all_gain - 0.01
        ),
        "P4_communication_causal": derived["disable_damage"] >= 0.02,
        "P5_source_causal": derived["source_shuffle_damage"] >= 0.50,
        "P6_multiresolution": sum(value >= 0.05 for value in route) >= 2,
        "P7_finite_nonempty": (
            butterfly["finite_gradients"]
            and math.isfinite(butterfly_nll)
            and butterfly["test"].get("nonempty", 1.0) > 0.0
        ),
    }
    return derived, gates


def mechanism_gates(seed_results: Dict[str, dict]):
    parameters = [row["parameters"] for row in seed_results.values()]
    return {
        "M1_finite_gradients": all(row["finite_gradients"] for row in seed_results.values()),
        "M2_initial_identity": all(
            row["initial_communication"]["initial_or_trained_delta_rms"] <= 1e-7
            for row in seed_results.values()
        ),
        "M3_communication_inverse": all(
            row["trained_communication"]["inverse_mse"] <= 1e-10
            for row in seed_results.values()
        ),
        "M4_fold_unfold_closure": all(
            row["closure"].get("state_mse", 1.0) <= 1e-10
            for row in seed_results.values()
        ),
        "M5_parameter_matched": len(set(parameters)) == 1,
        "M6_no_dense_attention": all(
            not row["trained_communication"]["dense_attention_allocated"]
            for row in seed_results.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s2_treeheap_butterfly_wmt_smoke")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--valid-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--max-scan", type=int)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--long-min", type=int, default=25)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--coupling-scale", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--generate-examples", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    defaults = {
        "smoke": dict(
            seeds=[8104], train_samples=5000, valid_samples=500,
            test_samples=500, max_scan=100000, dim=128, hidden=128, epochs=2,
        ),
        "formal": dict(
            seeds=[8104, 8105, 8106], train_samples=200000, valid_samples=5000,
            test_samples=5000, max_scan=2000000, dim=256, hidden=256, epochs=5,
        ),
    }[args.mode]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")
    if args.heap_width & (args.heap_width - 1):
        raise ValueError("heap width must be a power of two")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    all_results = {}
    sampling_rows = {}
    trace_rows = []

    for seed in args.seeds:
        args.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rows, sampling = adaptive.load_rows(args, sp)
        sampling_rows[str(seed)] = sampling
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        arm_results = {}
        for arm in ARMS:
            arm_results[arm] = train_arm(
                arm, seed, rows, args, vocab, pad, bos, eos, sp, seed_output,
            )
            trace_rows.extend(arm_results[arm]["trace"])
        derived, gates = seed_decision(arm_results)
        mechanism = mechanism_gates(arm_results)
        all_results[str(seed)] = {
            "models": arm_results,
            "derived": derived,
            "gates": gates,
            "mechanism_gates": mechanism,
            "language_pass": all(gates.values()),
            "mechanism_pass": all(mechanism.values()),
        }
        print(json.dumps({
            "seed": seed, "derived": derived, "gates": gates,
            "mechanism_gates": mechanism,
        }, indent=2), flush=True)

    seed_rows = list(all_results.values())
    pass_count = sum(row["language_pass"] for row in seed_rows)
    mechanism_pass = all(row["mechanism_pass"] for row in seed_rows)
    mean_nll = {
        arm: mean(row["models"][arm]["test"]["nll"] for row in seed_rows)
        for arm in ARMS
    }
    formal_supported = (
        args.mode == "formal"
        and pass_count >= 2
        and mean_nll["butterfly"] < min(mean_nll["identity"], mean_nll["adjacent"])
        and mechanism_pass
    )
    decision = (
        "smoke_pass" if args.mode == "smoke" and mechanism_pass
        else "smoke_fail" if args.mode == "smoke"
        else "supported" if formal_supported
        else "not_supported"
    )
    aggregate = {
        "language_seed_pass_count": pass_count,
        "mechanism_all_seeds_pass": mechanism_pass,
        "mean_test_nll": mean_nll,
        "butterfly_mean_gain_over_identity": mean_nll["identity"] - mean_nll["butterfly"],
        "butterfly_mean_gain_over_adjacent": mean_nll["adjacent"] - mean_nll["butterfly"],
    }
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "host": socket.gethostname(),
        "seconds": time.time() - started,
        "config": vars(args),
        "data": {
            "direction": "en_to_zh",
            "vocab": vocab,
            "sampling_by_seed": sampling_rows,
        },
        "seeds": all_results,
        "aggregate": aggregate,
        "decision": decision,
        "boundary": (
            "Matched WMT mechanism ablation. No Transformer superiority, "
            "production quality, semantic-address, or compute-saving claim."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace_rows) + "\n", encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# TreeHeap Butterfly WMT matched ablation\n\n```json\n"
        + json.dumps({"aggregate": aggregate, "decision": decision}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
