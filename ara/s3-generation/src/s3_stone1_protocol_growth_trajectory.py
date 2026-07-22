#!/usr/bin/env python3
"""Observe whether a 50M TreeHeap decoder grows recursive reads over training.

Owner: Nio Log Squad
Author: OpenAI Codex
Created: 2026-07-23
Updated: 2026-07-23
Purpose: Train one frozen-contract model continuously and measure quality,
route depth, pre-fold path causality, and post-fold detail causality.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_private_protocol_data_dose as data_dose
import s3_stone1_canonical_codec as c02
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


PARAMETERS = 50_267_778
FORMAL_MILESTONES = (15_625, 31_250, 46_875, 62_500)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--baseline-summary", default=(
        "/home/nio/log/holds/SameTime/ara/s3-generation/evidence/"
        "s3_stone1_canonical_codec/summary.json"
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
    parser.add_argument("--fixed-steps", type=int, default=62_500)
    parser.add_argument("--cli-max-new-tokens", type=int, default=64)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def milestone_audit(model, loader, args, pad, bos, eos, sp, step: int) -> dict:
    normal = c02.evaluate(model, loader, args, pad, bos, eos, sp)
    root = c02.evaluate(
        model, loader, args, pad, bos, eos, sp, max_visible_levels=1,
    )
    postfold = c02.evaluate(
        model, loader, args, pad, bos, eos, sp, intervention="address_swap",
    )
    algebraic = c02.evaluate(
        model, loader, args, pad, bos, eos, sp, codec_override="algebraic",
    )
    mirrors = []
    for depth in range(model.encoder.depths):
        score = c02.evaluate(
            model, loader, args, pad, bos, eos, sp, fold_mirror_depth=depth,
        )
        mirrors.append({
            "depth": depth, "nll": score["nll"],
            "damage_nll": score["nll"] - normal["nll"],
        })
    route = normal.get("route_mass_by_level", [])
    return {
        "step": step,
        "normal": normal,
        "root_only": root,
        "postfold_address_swap": postfold,
        "force_algebraic": algebraic,
        "prefold_mirrors": mirrors,
        "nonroot_route_mass": max(0.0, 1.0 - (route[0] if route else 1.0)),
        "root_to_full_gain_nll": root["nll"] - normal["nll"],
        "postfold_address_damage_nll": postfold["nll"] - normal["nll"],
        "force_algebraic_damage_nll": algebraic["nll"] - normal["nll"],
        "max_prefold_mirror_damage_nll": max(row["damage_nll"] for row in mirrors),
    }


def save_checkpoint(path: Path, model, optimizer, step: int, args, manifest: dict):
    payload = {
        "experiment_id": "s3_stone1_protocol_growth_trajectory",
        "step": step,
        "model_config": {
            "dim": args.dim, "hidden": args.hidden,
            "heap_width": args.heap_width, "leaf_cut": args.leaf_cut,
        },
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_hashes": manifest["splits"],
    }
    torch.save(payload, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": c01.file_digest(path)}


def decide(milestones: list[dict], final_test: dict) -> tuple[dict, str]:
    by_step = {row["step"]: row for row in milestones}
    first = by_step[min(by_step)]
    final = by_step[max(by_step)]
    later = milestones[1:]
    gates = {
        "G1_nll_gain_at_31250_at_least_0_08": (
            len(milestones) < 2
            or first["normal"]["nll"] - milestones[1]["normal"]["nll"] >= 0.08
        ),
        "G2_nonroot_route_growth": (
            any(row["nonroot_route_mass"] >= 0.10 for row in later)
            and final["nonroot_route_mass"] >= 0.05
        ),
        "G3_final_root_to_full_gain_at_least_0_10": final["root_to_full_gain_nll"] >= 0.10,
        "G4_final_postfold_address_damage_at_least_0_10": final["postfold_address_damage_nll"] >= 0.10,
        "G5_final_prefold_mirror_damage_at_least_0_10": final["max_prefold_mirror_damage_nll"] >= 0.10,
        "E1_finite_final_nll": math.isfinite(final_test["nll"]),
        "E2_generation_nonempty": final_test["nonempty"] == 1.0,
    }
    if len(milestones) < 4:
        status = "smoke_only"
    elif any(gates[key] for key in (
        "G2_nonroot_route_growth",
        "G3_final_root_to_full_gain_at_least_0_10",
        "G4_final_postfold_address_damage_at_least_0_10",
    )):
        status = "recursive_growth_signal_single_seed"
    elif gates["G5_final_prefold_mirror_damage_at_least_0_10"]:
        status = "path_sensitive_root_compression_single_seed"
    else:
        status = "bag_like_root_collapse_single_seed"
    return gates, status


def main() -> None:
    args = parse_args()
    milestones = list(FORMAL_MILESTONES)
    if args.smoke:
        args.train_samples = 4096
        args.valid_samples = 128
        args.test_samples = 128
        args.baseline_max_scan = 100_000
        args.pool_max_scan = 200_000
        args.fixed_steps = 1000
        args.eval_interval = 100
        milestones = [500, 1000]
    elif args.fixed_steps != FORMAL_MILESTONES[-1]:
        raise ValueError("formal C04 requires 62,500 continuous updates")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    config = vars(args).copy()
    config["milestones"] = milestones
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

    c01.set_seed(args.model_seed)
    model = c02.make_model("canonical_learned", args, vocab, pad).to(args.device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
        raise ValueError("registered 50M parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad, args.model_seed + args.train_samples,
    )
    trace_path = output / "trace.jsonl"
    audits: list[dict] = []
    checkpoints = []
    window_loss = 0.0
    window_steps = 0
    finite = True

    initial = c02.evaluate(model, valid_loader, args, pad, bos, eos, sp)
    data_dose.append_trace(trace_path, {
        "event": "evaluation", "step": 0, "train_nll_window": None,
        "valid_nll": initial["nll"],
        "route_mass_by_level": initial.get("route_mass_by_level", []),
    })
    for step in range(1, args.fixed_steps + 1):
        _, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        model.train()
        logits, _ = model.teacher(source, length, target, bos)
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1

        if step % args.eval_interval == 0 or step in milestones:
            valid_score = c02.evaluate(model, valid_loader, args, pad, bos, eos, sp)
            row = {
                "event": "evaluation", "step": step,
                "sample_exposures": step * args.batch_size,
                "train_nll_window": window_loss / max(1, window_steps),
                "valid_nll": valid_score["nll"],
                "route_mass_by_level": valid_score.get("route_mass_by_level", []),
                "elapsed_sec": time.time() - started,
            }
            data_dose.append_trace(trace_path, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = 0.0
            window_steps = 0

        if step in milestones:
            audit = milestone_audit(
                model, valid_loader, args, pad, bos, eos, sp, step,
            )
            audits.append(audit)
            checkpoint = save_checkpoint(
                checkpoint_dir / f"growth_step{step}.pt",
                model, optimizer, step, args, manifest,
            )
            checkpoints.append(checkpoint)
            (output / "milestones.json").write_text(
                json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(json.dumps({"event": "milestone", **audit}, ensure_ascii=False), flush=True)

    final_test = c02.evaluate(
        model, test_loader, args, pad, bos, eos, sp, generate=True,
    )
    gates, status = decide(audits, final_test)
    summary = {
        "experiment_id": "s3_stone1_protocol_growth_trajectory",
        "claim": "S3-STONE1-PROTOCOL-GROWTH-C04",
        "predict": "P-S3-STONE1-PROTOCOL-GROWTH-04",
        "status": status,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "git_commit": args.code_commit or c01.git_revision(),
        "seconds": time.time() - started,
        "parameters": PARAMETERS,
        "finite_gradients": finite,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        "platform_checks": checks,
        "dataset": manifest,
        "milestones": audits,
        "final_test": final_test,
        "gates": gates,
        "checkpoints": checkpoints,
        "boundary": (
            "Single-seed trajectory separates recursive decoder growth from "
            "path-sensitive root compression; it does not establish stable emergence."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": status, "seconds": summary["seconds"],
        "final_test_nll": final_test["nll"], "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
