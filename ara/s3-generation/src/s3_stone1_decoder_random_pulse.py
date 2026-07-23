#!/usr/bin/env python3
"""Test whether temporary forced-depth pulses accumulate or overwrite."""
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_lifting_pump_wmt as prior
import s3_private_protocol_data_dose as data_dose
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


FORMAL_STEPS = 15_625
FORMAL_ARMS = {
    "random_1": ("random", 1),
    "random_32": ("random", 32),
    "random_256": ("random", 256),
    "cyclic_32": ("cyclic", 32),
}
SMOKE_ARMS = {
    "random_1": ("random", 1),
    "random_8": ("random", 8),
    "random_32": ("random", 32),
    "cyclic_8": ("cyclic", 8),
}


class PulseDecoder(prior.RecursiveDecoder):
    """Read one exact depth or an equal mixture of all depth contexts."""

    def _contexts(self, hidden: torch.Tensor, levels, masks):
        active = torch.ones((hidden.shape[0], 1), device=hidden.device)
        contexts = []
        for depth, (nodes, valid) in enumerate(zip(levels, masks)):
            active = active * valid.to(active.dtype)
            normalizer = active.sum(1, keepdim=True).clamp_min(1e-8)
            contexts.append((active[:, :, None] * nodes).sum(1) / normalizer)
            if depth == len(levels) - 1:
                continue
            children = levels[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2]
            )
            child_valid = masks[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2
            )
            scores = (
                self.branch(hidden)[:, None, None] * children
            ).sum(-1) / math.sqrt(nodes.shape[-1])
            probability = F.softmax(
                scores.masked_fill(~child_valid, -1e9), dim=-1
            )
            active = (active[:, :, None] * probability).reshape(
                nodes.shape[0], -1
            )
        return contexts

    def read(self, hidden: torch.Tensor, levels, masks, route_mode: str = "native"):
        if len(levels) != self.depths:
            raise ValueError(
                f"decoder expected {self.depths} visible levels, got {len(levels)}"
            )
        if route_mode == "uniform_depth":
            contexts = self._contexts(hidden, levels, masks)
            weights = hidden.new_full((len(contexts),), 1.0 / len(contexts))
            return torch.stack(contexts, dim=1).mean(1), weights
        if route_mode.startswith("force_depth_"):
            target = int(route_mode.rsplit("_", 1)[1])
            if not 0 <= target < len(levels):
                raise ValueError(f"invalid forced depth {target}")
            contexts = self._contexts(hidden, levels, masks)
            weights = hidden.new_zeros((len(contexts),))
            weights[target] = 1.0
            return contexts[target], weights
        return super().read(hidden, levels, masks, route_mode)


class PulseSchedule:
    def __init__(self, mode: str, block: int, depths: int, seed: int):
        self.mode = mode
        self.block = block
        self.depths = depths
        self.rng = random.Random(seed)
        self.current_block = -1
        self.current_depth = 0

    def depth(self, step: int) -> int:
        block_index = (step - 1) // self.block
        if block_index != self.current_block:
            self.current_block = block_index
            if self.mode == "random":
                self.current_depth = self.rng.randrange(self.depths)
            elif self.mode == "cyclic":
                self.current_depth = block_index % self.depths
            else:
                raise ValueError(self.mode)
        return self.current_depth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--baseline-summary", default=(
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_canonical_codec/summary.json"
    ))
    parser.add_argument("--c04-checkpoint", default=(
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_protocol_growth_trajectory/checkpoints/growth_step62500.pt"
    ))
    parser.add_argument("--c06-summary", default=(
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_decoder_depth_floor/summary.json"
    ))
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
    parser.add_argument("--probe-samples", type=int, default=256)
    parser.add_argument("--baseline-max-scan", type=int, default=300_000)
    parser.add_argument("--pool-max-scan", type=int, default=3_000_000)
    parser.add_argument("--source-rows", type=int, default=14_170_275)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71900)
    parser.add_argument("--pool-seed", type=int, default=72003)
    parser.add_argument("--model-seed", type=int, default=71902)
    parser.add_argument("--schedule-seed", type=int, default=73103)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--gradient-interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--fixed-steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_model(args, vocab: int, pad: int):
    model = c05.load_model(args, vocab, pad)
    replacement = PulseDecoder(
        vocab, args.dim, args.hidden, model.encoder.depths,
    )
    replacement.load_state_dict(model.decoder.state_dict(), strict=True)
    model.decoder = replacement
    return model


def forced_depth_vector(model, loader, args, pad, bos, eos, sp):
    rows = []
    for depth in range(model.encoder.depths):
        score = c05.evaluate(
            model, loader, args, pad, bos, eos, sp, f"force_depth_{depth}",
        )
        rows.append(score["nll"])
    return rows


def gradient_cosine_diagnostic(
    model, batch, args, pad: int, bos: int,
):
    source, length, target, _ = batch
    source = source.to(args.device, non_blocking=True)
    length = length.to(args.device, non_blocking=True)
    target = target.to(args.device, non_blocking=True)
    parameters = [
        model.decoder.cell.weight_ih,
        model.decoder.cell.weight_hh,
    ]
    vectors = []
    model.train()
    for depth in range(model.encoder.depths):
        model.zero_grad(set_to_none=True)
        logits, _ = model.teacher(
            source, length, target, bos,
            route_mode=f"force_depth_{depth}",
        )
        loss = base.ce(logits, target, pad)
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=False,
        )
        vectors.append(torch.cat([row.detach().flatten() for row in gradients]))
    matrix = []
    off_diagonal = []
    for left, first in enumerate(vectors):
        row = []
        for right, second in enumerate(vectors):
            value = float(F.cosine_similarity(first, second, dim=0))
            row.append(value)
            if left < right:
                off_diagonal.append(value)
        matrix.append(row)
    model.zero_grad(set_to_none=True)
    return {
        "matrix": matrix,
        "mean_off_diagonal": sum(off_diagonal) / len(off_diagonal),
        "negative_fraction": (
            sum(value < 0.0 for value in off_diagonal) / len(off_diagonal)
        ),
        "minimum": min(off_diagonal),
        "maximum": max(off_diagonal),
    }


def train_arm(
    arm: str, schedule_spec, args, rows, valid_loader, test_loader, probe_loader,
    probe_batch, vocab: int, pad: int, bos: int, eos: int, sp, output: Path,
):
    c01.set_seed(args.model_seed)
    model = load_model(args, vocab, pad).to(args.device)
    encoder_before = c05.tensor_digest(model.encoder)
    trainable = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad,
        args.model_seed + args.train_samples,
    )
    depths = model.encoder.depths
    schedule = PulseSchedule(
        schedule_spec[0], schedule_spec[1], depths,
        args.schedule_seed + sum(ord(char) for char in arm),
    )
    initial_probe = forced_depth_vector(
        model, probe_loader, args, pad, bos, eos, sp,
    )
    best_probe = list(initial_probe)
    trace = []
    route_counts = [0] * depths
    branch_nonzero = branch_observations = 0
    finite = True
    started = time.time()
    window_loss = 0.0
    window_steps = 0

    for step in range(1, args.fixed_steps + 1):
        batch = next(batches)
        _, (source, length, target, _) = batch
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        depth = schedule.depth(step)
        route_counts[depth] += 1
        model.train()
        logits, _ = model.teacher(
            source, length, target, bos,
            route_mode=f"force_depth_{depth}",
        )
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if depth > 0:
            gradient = model.decoder.branch.weight.grad
            if gradient is not None:
                branch_observations += 1
                branch_nonzero += int(float(gradient.detach().abs().max()) > 0.0)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1

        if step % args.eval_interval == 0 or step == args.fixed_steps:
            probe = forced_depth_vector(
                model, probe_loader, args, pad, bos, eos, sp,
            )
            best_probe = [
                min(best, current)
                for best, current in zip(best_probe, probe)
            ]
            forgetting = [
                current - best
                for current, best in zip(probe, best_probe)
            ]
            diagnostic = None
            if step % args.gradient_interval == 0 or step == args.fixed_steps:
                diagnostic = gradient_cosine_diagnostic(
                    model, probe_batch, args, pad, bos,
                )
            row = {
                "arm": arm,
                "step": step,
                "active_depth": depth,
                "train_nll_window": window_loss / max(1, window_steps),
                "probe_nll_by_depth": probe,
                "forgetting_by_depth": forgetting,
                "max_forgetting": max(forgetting),
                "route_update_fraction": [
                    count / step for count in route_counts
                ],
                "gradient_cosine": diagnostic,
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = 0.0
            window_steps = 0

    final_valid = forced_depth_vector(
        model, valid_loader, args, pad, bos, eos, sp,
    )
    final_test = forced_depth_vector(
        model, test_loader, args, pad, bos, eos, sp,
    )
    uniform_test = c05.evaluate(
        model, test_loader, args, pad, bos, eos, sp,
        "uniform_depth", generate=True,
    )
    final_probe = trace[-1]["probe_nll_by_depth"]
    final_forgetting = trace[-1]["forgetting_by_depth"]
    gains = [
        initial - final
        for initial, final in zip(initial_probe, final_probe)
    ]
    encoder_after = c05.tensor_digest(model.encoder)
    checkpoint = output / "checkpoints" / f"decoder_{arm}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "arm": arm,
        "schedule": schedule_spec,
        "decoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.decoder.state_dict().items()
        },
        "source_checkpoint": args.c04_checkpoint,
    }, checkpoint)
    return {
        "arm": arm,
        "schedule": {"mode": schedule_spec[0], "block": schedule_spec[1]},
        "initial_probe_nll_by_depth": initial_probe,
        "final_probe_nll_by_depth": final_probe,
        "final_valid_nll_by_depth": final_valid,
        "final_test_nll_by_depth": final_test,
        "uniform_test": uniform_test,
        "probe_gain_by_depth": gains,
        "mean_probe_gain": sum(gains) / len(gains),
        "improved_depths_at_0_10": sum(gain >= 0.10 for gain in gains),
        "final_forgetting_by_depth": final_forgetting,
        "max_final_forgetting": max(final_forgetting),
        "max_observed_forgetting": max(
            row["max_forgetting"] for row in trace
        ),
        "route_update_count": route_counts,
        "route_update_fraction": [
            count / args.fixed_steps for count in route_counts
        ],
        "branch_grad_nonzero_fraction": (
            branch_nonzero / max(1, branch_observations)
        ),
        "finite_gradients": finite,
        "encoder_unchanged": encoder_before == encoder_after,
        "trace": trace,
        "seconds": time.time() - started,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": c01.file_digest(checkpoint),
        },
    }


def main():
    args = parse_args()
    arms_spec = FORMAL_ARMS
    if args.smoke:
        args.train_samples = 4096
        args.valid_samples = 128
        args.test_samples = 128
        args.probe_samples = 64
        args.baseline_max_scan = 100_000
        args.pool_max_scan = 200_000
        args.fixed_steps = 240
        args.eval_interval = 80
        args.gradient_interval = 80
        arms_spec = SMOKE_ARMS
    elif args.fixed_steps != FORMAL_STEPS:
        raise ValueError(f"formal C07 requires {FORMAL_STEPS} updates per arm")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    config = vars(args).copy()
    config["arms"] = arms_spec
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    started = time.time()
    baseline = c03.load_baseline(args.baseline_summary)
    c06 = json.loads(Path(args.c06_summary).read_text(encoding="utf-8"))
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    checks = c03.verify_frozen_platform(manifest, baseline, args.smoke)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)
    probe = valid[: min(args.probe_samples, len(valid))]
    probe_loader = data_dose.make_loader(probe, args, pad, False)
    probe_batch = next(iter(probe_loader))
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    arms = {}
    for arm, schedule_spec in arms_spec.items():
        arms[arm] = train_arm(
            arm, schedule_spec, args, rows, valid_loader, test_loader,
            probe_loader, probe_batch, vocab, pad, bos, eos, sp, output,
        )
        (output / "partial_summary.json").write_text(
            json.dumps(arms, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    primary_name = "random_32"
    primary = arms[primary_name]
    coverage_threshold = 0.08 if not args.smoke else 0.0
    gain_threshold = 0.20 if not args.smoke else 0.0
    improved_threshold = 5 if not args.smoke else 0
    forgetting_threshold = 0.15 if not args.smoke else float("inf")
    gates = {
        "G1_every_depth_update_fraction_at_least_0_08": (
            min(primary["route_update_fraction"]) >= coverage_threshold
        ),
        "G2_mean_probe_nll_gain_at_least_0_20": (
            primary["mean_probe_gain"] >= gain_threshold
        ),
        "G3_at_least_five_depths_gain_0_10": (
            primary["improved_depths_at_0_10"] >= improved_threshold
        ),
        "G4_max_final_forgetting_at_most_0_15": (
            primary["max_final_forgetting"] <= forgetting_threshold
        ),
        "G5_branch_gradients_finite_and_nonzero": (
            primary["finite_gradients"]
            and primary["branch_grad_nonzero_fraction"] > 0.99
        ),
        "G6_encoder_unchanged": all(
            row["encoder_unchanged"] for row in arms.values()
        ),
    }
    supported = all(gates.values())
    summary = {
        "experiment_id": "s3_stone1_decoder_random_pulse",
        "claim": "S3-STONE1-DECODER-RANDOM-PULSE-C07",
        "predict": "P-S3-STONE1-DECODER-RANDOM-PULSE-07",
        "status": (
            "smoke_completed" if args.smoke else
            "random_pulse_accumulation_supported_single_seed"
            if supported else
            "random_pulse_accumulation_not_supported_under_recipe"
        ),
        "host": socket.gethostname(),
        "device_name": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
        ),
        "git_commit": args.code_commit or c01.git_revision(),
        "seconds": time.time() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda") else 0
        ),
        "platform_checks": checks,
        "dataset": manifest,
        "source_checkpoint": args.c04_checkpoint,
        "c06_reference": {
            "status": c06["status"],
            "depth_floor": c06["depth_floor"],
            "test_nll": c06["arms"]["depth_floor"]["final_test"]["nll"],
        },
        "primary_arm": primary_name,
        "arms": arms,
        "gates": gates,
        "boundary": (
            "Temporary depth pulses test parameter accumulation versus "
            "overwriting. Passing does not establish spontaneous routing, "
            "subjective private protocol, or STONE-1 completion."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "primary_mean_probe_gain": primary["mean_probe_gain"],
        "primary_max_final_forgetting": primary["max_final_forgetting"],
        "primary_update_fraction": primary["route_update_fraction"],
        "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
