#!/usr/bin/env python3
"""Test learned suppression of regular and irregular tails under a fixed root."""
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_private_protocol_data_dose as data_dose
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


FORMAL_STEPS = 15_625
TRAIN_ARMS = ("clean_mask", "eos_tail", "random_tail")
EVAL_CONDITIONS = ("clean_mask", "masked_random", "eos_tail", "random_tail")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv",
    )
    parser.add_argument(
        "--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
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
    parser.add_argument("--noise-seed", type=int, default=74231)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--fixed-steps", type=int, default=FORMAL_STEPS)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def fixed_source(
    source: torch.Tensor,
    length: torch.Tensor,
    condition: str,
    heap_width: int,
    pad: int,
    eos: int,
    pieces: int,
    noise_seed: int,
):
    if source.shape[1] > heap_width:
        raise ValueError(
            f"source width {source.shape[1]} exceeds heap width {heap_width}"
        )
    batch = source.shape[0]
    fixed = torch.full(
        (batch, heap_width), pad, dtype=source.dtype, device=source.device,
    )
    fixed[:, : source.shape[1]] = source
    position = torch.arange(heap_width, device=source.device)[None]
    tail = position >= length[:, None]

    if condition in {"masked_random", "random_tail"}:
        weights = torch.arange(
            1, source.shape[1] + 1, device=source.device, dtype=torch.long,
        )
        signature = (
            source.long() * (weights[None] * 104729)
        ).sum(1, keepdim=True)
        noise = (
            signature * 1103515245
            + position.long() * 123457
            + noise_seed
        ).remainder(pieces)
        fixed = torch.where(tail, noise.to(fixed.dtype), fixed)
    elif condition == "eos_tail":
        fixed = torch.where(tail, torch.full_like(fixed, eos), fixed)
    elif condition != "clean_mask":
        raise ValueError(condition)

    visible_length = (
        torch.full_like(length, heap_width)
        if condition in {"eos_tail", "random_tail"}
        else length
    )
    return fixed, visible_length


@torch.no_grad()
def evaluate(
    model, loader, args, pad: int, bos: int, eos: int, pieces: int,
    condition: str, generate: bool = False, sp=None,
):
    model.eval()
    loss_sum = tokens = count = repeated = nonempty = 0
    route_sum = None
    route_batches = 0
    hypotheses, references, examples = [], [], []
    for source, length, target, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        source, visible_length = fixed_source(
            source, length, condition, args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        logits, route = model.teacher(
            source, visible_length, target, bos, route_mode="depth_floor",
        )
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
        route_cpu = route.detach().float().cpu()
        route_sum = route_cpu if route_sum is None else route_sum + route_cpu
        route_batches += 1
        if not generate:
            continue
        predicted, _ = model.greedy(
            source, visible_length, bos, eos, target.shape[1],
            route_mode="depth_floor",
        )
        for index in range(source.shape[0]):
            hyp = base.clean(predicted[index].cpu().tolist(), eos, pad)
            ref = base.clean(target[index].cpu().tolist(), eos, pad)
            hypotheses.append(hyp)
            references.append(ref)
            nonempty += int(bool(hyp))
            repeated += int(c01.severe_repetition(hyp))
            count += 1
            if len(examples) < 8 and sp is not None:
                original = base.clean(
                    source[index, : int(length[index])].cpu().tolist(), eos, pad,
                )
                examples.append({
                    "en": sp.decode(original),
                    "reference_zh": sp.decode(ref),
                    "hypothesis_zh": sp.decode(hyp),
                })
    nll = loss_sum / max(1, tokens)
    result = {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": tokens,
        "route_mass_by_level": (
            route_sum / max(1, route_batches)
        ).tolist(),
    }
    if generate:
        result.update({
            "token_bleu4": base.bleu4(hypotheses, references),
            "nonempty": nonempty / max(1, count),
            "severe_repetition_rate": repeated / max(1, count),
            "examples": examples,
        })
    return result


@torch.no_grad()
def root_geometry(model, loader, args, pad, eos, pieces):
    model.eval()
    sums = {condition: 0.0 for condition in EVAL_CONDITIONS}
    counts = 0
    for source, length, _, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        roots = {}
        for condition in EVAL_CONDITIONS:
            changed, visible_length = fixed_source(
                source, length, condition, args.heap_width, pad, eos, pieces,
                args.noise_seed,
            )
            roots[condition] = model.states(changed, visible_length)[1]
        clean = roots["clean_mask"]
        for condition in EVAL_CONDITIONS:
            sums[condition] += float(
                F.cosine_similarity(clean, roots[condition], dim=-1).sum()
            )
        counts += source.shape[0]
    return {
        condition: value / max(1, counts)
        for condition, value in sums.items()
    }


def train_arm(
    arm, args, rows, valid_loader, test_loader, vocab, pieces,
    pad, bos, eos, sp, output,
):
    c01.set_seed(args.model_seed)
    model = c06.load_model(args, vocab, pad).to(args.device)
    encoder_before = c05.tensor_digest(model.encoder)
    trainable = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad,
        args.model_seed + args.train_samples,
    )
    trace = []
    finite = True
    window_loss = 0.0
    started = time.time()
    for step in range(1, args.fixed_steps + 1):
        _, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        source, visible_length = fixed_source(
            source, length, arm, args.heap_width, pad, eos, pieces,
            args.noise_seed,
        )
        logits, _ = model.teacher(
            source, visible_length, target, bos, route_mode="depth_floor",
        )
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        if step % args.eval_interval == 0 or step == args.fixed_steps:
            valid = evaluate(
                model, valid_loader, args, pad, bos, eos, pieces, arm,
            )
            row = {
                "arm": arm,
                "step": step,
                "train_nll_window": window_loss / (
                    args.eval_interval
                    if step % args.eval_interval == 0
                    else step % args.eval_interval
                ),
                "matched_valid_nll": valid["nll"],
                "route_mass_by_level": valid["route_mass_by_level"],
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = 0.0

    cross_valid = {
        condition: evaluate(
            model, valid_loader, args, pad, bos, eos, pieces, condition,
        )
        for condition in EVAL_CONDITIONS
    }
    cross_test = {
        condition: evaluate(
            model, test_loader, args, pad, bos, eos, pieces, condition,
        )
        for condition in EVAL_CONDITIONS
    }
    matched_generation = evaluate(
        model, test_loader, args, pad, bos, eos, pieces, arm, True, sp,
    )
    encoder_after = c05.tensor_digest(model.encoder)
    checkpoint = output / "checkpoints" / f"decoder_{arm}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({
        "arm": arm,
        "decoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.decoder.state_dict().items()
        },
        "source_checkpoint": args.c04_checkpoint,
    }, checkpoint)
    return {
        "arm": arm,
        "trace": trace,
        "cross_valid": cross_valid,
        "cross_test": cross_test,
        "matched_generation": matched_generation,
        "finite_gradients": finite,
        "encoder_unchanged": encoder_before == encoder_after,
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
        args.fixed_steps = 240
        args.eval_interval = 80
    elif args.fixed_steps != FORMAL_STEPS:
        raise ValueError(f"formal C08 requires {FORMAL_STEPS} updates per arm")
    if args.heap_width != 64:
        raise ValueError("C08 preregisters a fixed 64-leaf TreeHeap")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.base_train_samples = min(30_000, args.train_samples)
    args.doses = sorted(set((args.base_train_samples, args.train_samples)))
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8",
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

    c01.set_seed(args.model_seed)
    initial_model = c06.load_model(args, vocab, pad).to(args.device)
    initial = {
        condition: evaluate(
            initial_model, valid_loader, args, pad, bos, eos, pieces, condition,
        )
        for condition in EVAL_CONDITIONS
    }
    geometry = root_geometry(
        initial_model, valid_loader, args, pad, eos, pieces,
    )
    del initial_model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    arms = {}
    for arm in TRAIN_ARMS:
        arms[arm] = train_arm(
            arm, args, rows, valid_loader, test_loader, vocab, pieces,
            pad, bos, eos, sp, output,
        )
        (output / "partial_summary.json").write_text(
            json.dumps(arms, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    clean_initial = initial["clean_mask"]["nll"]
    masked_initial = initial["masked_random"]["nll"]
    eos_initial = initial["eos_tail"]["nll"]
    eos_final = arms["eos_tail"]["cross_valid"]["eos_tail"]["nll"]
    random_final = arms["random_tail"]["cross_valid"]["random_tail"]["nll"]
    eos_clean = arms["eos_tail"]["cross_valid"]["clean_mask"]["nll"]
    clean_final = arms["clean_mask"]["cross_valid"]["clean_mask"]["nll"]
    gates = {
        "G1_masked_random_is_numerically_invariant": (
            abs(clean_initial - masked_initial) <= 1e-7
        ),
        "G2_eos_matched_nll_gain_at_least_0_30": (
            eos_initial - eos_final >= 0.30
        ),
        "G3_eos_beats_random_matched_nll_by_0_10": (
            random_final - eos_final >= 0.10
        ),
        "G4_eos_training_retains_clean_within_0_15": (
            eos_clean - clean_final <= 0.15
        ),
        "G5_eos_measurably_changes_frozen_root": (
            geometry["eos_tail"] < 0.995
        ),
        "G6_encoder_frozen_and_gradients_finite": all(
            arm["encoder_unchanged"] and arm["finite_gradients"]
            for arm in arms.values()
        ),
    }
    supported = all(gates.values())
    summary = {
        "experiment_id": "s3_stone1_fixed_root_noise_repair",
        "claim": "S3-STONE1-FIXED-ROOT-NOISE-REPAIR-C08",
        "predict": "P-S3-STONE1-FIXED-ROOT-NOISE-REPAIR-08",
        "status": (
            "smoke_completed" if args.smoke else
            "regular_noise_repair_supported_single_seed"
            if supported else "regular_noise_repair_not_supported_under_recipe"
        ),
        "host": socket.gethostname(),
        "device_name": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda") else "cpu"
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
        "fixed_root_contract": {
            "heap_width": args.heap_width,
            "root_moved": False,
            "encoder_frozen": True,
            "decoder_route": "depth_floor",
            "depth_floor": c06.DEPTH_FLOOR,
        },
        "initial_valid": initial,
        "root_cosine_to_clean": geometry,
        "arms": arms,
        "gates": gates,
        "boundary": (
            "This tests decoder-side suppression of visible fixed-root tail "
            "noise. It does not prove protocol migration or persistent-memory "
            "repair."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "initial_nll": {
            key: value["nll"] for key, value in initial.items()
        },
        "root_cosine_to_clean": geometry,
        "matched_final_nll": {
            arm: value["cross_valid"][arm]["nll"]
            for arm, value in arms.items()
        },
        "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
