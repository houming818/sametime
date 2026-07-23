#!/usr/bin/env python3
"""Compare native, deepest, and bounded-pressure depth reads over frozen C04."""
from __future__ import annotations

import argparse
import copy
import json
import math
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
import s3_stone1_canonical_codec as c02
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


FORMAL_STEPS = 15_625
DEPTH_FLOOR = 0.02
ARMS = {
    "native_control": "native",
    "leaf_reference": "force_leaf",
    "depth_floor": "depth_floor",
}


class FloorPressureDecoder(prior.RecursiveDecoder):
    """Keep a fixed probability floor while learning depth allocation."""

    def __init__(self, vocab: int, dim: int, hidden: int, depths: int, floor: float):
        super().__init__(vocab, dim, hidden, depths)
        self.floor = floor

    def read(self, hidden: torch.Tensor, levels, masks, route_mode: str = "native"):
        if route_mode != "depth_floor":
            return super().read(hidden, levels, masks, route_mode)
        count = len(levels)
        if self.floor * count >= 1.0:
            raise ValueError("depth floor leaves no learnable route mass")

        query = self.query(hidden)
        active = torch.ones((hidden.shape[0], 1), device=hidden.device)
        contexts = []
        scores = []
        for depth, (nodes, valid) in enumerate(zip(levels, masks)):
            active = active * valid.to(active.dtype)
            normalizer = active.sum(1, keepdim=True).clamp_min(1e-8)
            contexts.append(
                (active[:, :, None] * nodes).sum(1) / normalizer
            )
            depth_state = self.depth_embedding.weight[depth][None, None]
            q = query[:, None].expand_as(nodes)
            raw = self.stop(
                torch.cat((q, nodes + depth_state), dim=-1)
            ).squeeze(-1)
            scores.append(
                torch.tanh((active * raw).sum(1) / normalizer.squeeze(1))
            )
            if depth == count - 1:
                continue
            children = levels[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2]
            )
            child_valid = masks[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2
            )
            branch_score = (
                self.branch(hidden)[:, None, None] * children
            ).sum(-1) / math.sqrt(nodes.shape[-1])
            probability = F.softmax(
                branch_score.masked_fill(~child_valid, -1e9), dim=-1
            )
            active = (active[:, :, None] * probability).reshape(
                nodes.shape[0], -1
            )

        learned = F.softmax(torch.stack(scores, dim=1), dim=1)
        weights = (1.0 - count * self.floor) * learned + self.floor
        stacked = torch.stack(contexts, dim=1)
        context = (weights[:, :, None] * stacked).sum(1)
        return context, weights.mean(0)


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
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--valid-samples", type=int, default=2_000)
    parser.add_argument("--test-samples", type=int, default=2_000)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=500)
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
    replacement = FloorPressureDecoder(
        vocab, args.dim, args.hidden, model.encoder.depths, DEPTH_FLOOR,
    )
    replacement.load_state_dict(model.decoder.state_dict(), strict=True)
    model.decoder = replacement
    return model


def train_arm(
    arm: str, route_mode: str, args, rows, valid_loader, test_loader,
    vocab: int, pad: int, bos: int, eos: int, sp, output: Path,
):
    c01.set_seed(args.model_seed)
    model = load_model(args, vocab, pad).to(args.device)
    encoder_before = c05.tensor_digest(model.encoder)
    trainable = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad, args.model_seed + args.train_samples,
    )
    initial = c05.evaluate(
        model, valid_loader, args, pad, bos, eos, sp, route_mode,
    )
    trace = []
    branch_nonzero = branch_observations = 0
    finite = True
    started = time.time()
    window_loss = 0.0
    window_steps = 0

    for step in range(1, args.fixed_steps + 1):
        _, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        model.train()
        logits, _ = model.teacher(
            source, length, target, bos, route_mode=route_mode,
        )
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        gradient = model.decoder.branch.weight.grad
        if gradient is not None:
            branch_observations += 1
            branch_nonzero += int(float(gradient.detach().abs().max()) > 0.0)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1

        if step % args.eval_interval == 0 or step == args.fixed_steps:
            valid = c05.evaluate(
                model, valid_loader, args, pad, bos, eos, sp, route_mode,
            )
            row = {
                "arm": arm, "step": step,
                "train_nll_window": window_loss / max(1, window_steps),
                "valid_nll": valid["nll"],
                "route_mass_by_level": valid["route_mass_by_level"],
                "branch_grad_nonzero_fraction": (
                    branch_nonzero / max(1, branch_observations)
                ),
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = window_steps = 0

    final_valid = c05.evaluate(
        model, valid_loader, args, pad, bos, eos, sp, route_mode,
    )
    final_test = c05.evaluate(
        model, test_loader, args, pad, bos, eos, sp, route_mode, generate=True,
    )
    detail_rows = []
    for depth in range(model.encoder.depths):
        score = c05.evaluate(
            model, test_loader, args, pad, bos, eos, sp, route_mode,
            intervention=f"detail_shuffle_{depth}",
        )
        detail_rows.append({
            "depth": depth,
            "nll": score["nll"],
            "damage_nll": score["nll"] - final_test["nll"],
        })
    encoder_after = c05.tensor_digest(model.encoder)
    checkpoint = output / "checkpoints" / f"decoder_{arm}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "arm": arm, "route_mode": route_mode,
        "decoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.decoder.state_dict().items()
        },
        "source_checkpoint": args.c04_checkpoint,
    }, checkpoint)
    return {
        "arm": arm,
        "route_mode": route_mode,
        "initial_valid": initial,
        "final_valid": final_valid,
        "final_test": final_test,
        "detail_shuffle": detail_rows,
        "max_detail_shuffle_damage_nll": max(
            row["damage_nll"] for row in detail_rows
        ),
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
    if args.smoke:
        args.train_samples = 4096
        args.valid_samples = 128
        args.test_samples = 128
        args.baseline_max_scan = 100_000
        args.pool_max_scan = 200_000
        args.fixed_steps = 200
        args.eval_interval = 100
    elif args.fixed_steps != FORMAL_STEPS:
        raise ValueError(f"formal C06 requires {FORMAL_STEPS} updates per arm")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    config = vars(args).copy()
    config["depth_floor"] = DEPTH_FLOOR
    config["arms"] = ARMS
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
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = data_dose.build_nested_data(args, sp, output)
    checks = c03.verify_frozen_platform(manifest, baseline, args.smoke)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = data_dose.make_loader(valid, args, pad, False)
    test_loader = data_dose.make_loader(test, args, pad, False)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    arms = {}
    for arm, route_mode in ARMS.items():
        arms[arm] = train_arm(
            arm, route_mode, args, rows, valid_loader, test_loader,
            vocab, pad, bos, eos, sp, output,
        )
        (output / "partial_summary.json").write_text(
            json.dumps(arms, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    floor = arms["depth_floor"]
    native = arms["native_control"]
    leaf = arms["leaf_reference"]
    route = floor["final_test"]["route_mass_by_level"]
    gates = {
        "G1_floor_branch_gradient_nonzero": (
            floor["finite_gradients"]
            and floor["branch_grad_nonzero_fraction"] > 0.99
        ),
        "G2_every_depth_mass_at_least_0_019": min(route) >= 0.019,
        "G3_floor_within_0_10_of_native": (
            floor["final_test"]["nll"] <= native["final_test"]["nll"] + 0.10
        ),
        "G4_floor_within_0_10_of_leaf": (
            floor["final_test"]["nll"] <= leaf["final_test"]["nll"] + 0.10
        ),
        "G5_detail_damage_at_least_0_10": (
            floor["max_detail_shuffle_damage_nll"] >= 0.10
        ),
        "G6_encoder_unchanged": all(
            row["encoder_unchanged"] for row in arms.values()
        ),
    }
    supported = all(gates.values())
    summary = {
        "experiment_id": "s3_stone1_decoder_depth_floor",
        "claim": "S3-STONE1-DECODER-DEPTH-FLOOR-C06",
        "predict": "P-S3-STONE1-DECODER-DEPTH-FLOOR-06",
        "status": (
            "bounded_pressure_learnable_depth_supported_single_seed"
            if supported else
            "bounded_pressure_learnable_depth_not_supported_under_recipe"
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
        "depth_floor": DEPTH_FLOOR,
        "arms": arms,
        "floor_minus_native_test_nll": (
            floor["final_test"]["nll"] - native["final_test"]["nll"]
        ),
        "floor_minus_leaf_test_nll": (
            floor["final_test"]["nll"] - leaf["final_test"]["nll"]
        ),
        "gates": gates,
        "boundary": (
            "A fixed depth floor is an architectural pressure supply; passing "
            "does not show the floor can be removed or STONE-1 is complete."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "floor_minus_native_test_nll": summary["floor_minus_native_test_nll"],
        "floor_minus_leaf_test_nll": summary["floor_minus_leaf_test_nll"],
        "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
