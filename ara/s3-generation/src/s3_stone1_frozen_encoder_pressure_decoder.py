#!/usr/bin/env python3
"""Train root-only and mandatory-recursive decoders over a frozen C04 encoder."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
import s3_stone1_canonical_codec as c02
import s3_stone1_capacity_rate_distortion as c03
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


ARMS = {"root_control": "force_root", "leaf_pressure": "force_leaf"}
FORMAL_STEPS = 15_625


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


def tensor_digest(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(args, vocab: int, pad: int):
    payload = torch.load(args.c04_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("step") != 62_500:
        raise ValueError("C05 requires the C04 update-62500 checkpoint")
    model = c02.make_model("canonical_learned", args, vocab, pad)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate(
    model, loader, args, pad: int, bos: int, eos: int, sp,
    route_mode: str, generate: bool = False, intervention: str = "native",
):
    model.eval()
    loss_sum = tokens = exact = nonempty = repeated = count = 0
    route_sum = None
    route_batches = 0
    hypotheses, references, examples = [], [], []
    for source, length, target, _ in loader:
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        logits, route = model.teacher(
            source, length, target, bos,
            route_mode=route_mode, intervention=intervention,
        )
        valid = target.ne(pad)
        loss_sum += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ))
        tokens += int(valid.sum())
        route_cpu = route.detach().float().cpu()
        route_sum = route_cpu if route_sum is None else route_sum + route_cpu
        route_batches += 1
        if not generate:
            continue
        predicted, _ = model.greedy(
            source, length, bos, eos, target.shape[1],
            route_mode=route_mode, intervention=intervention,
        )
        for index in range(source.shape[0]):
            hyp = base.clean(predicted[index].cpu().tolist(), eos, pad)
            ref = base.clean(target[index].cpu().tolist(), eos, pad)
            src = base.clean(source[index].cpu().tolist(), eos, pad)
            hypotheses.append(hyp)
            references.append(ref)
            exact += int(hyp == ref)
            nonempty += int(bool(hyp))
            repeated += int(c01.severe_repetition(hyp))
            count += 1
            if len(examples) < 12:
                examples.append({
                    "en": sp.decode(src),
                    "reference_zh": sp.decode(ref),
                    "hypothesis_zh": sp.decode(hyp),
                    "severe_repetition": c01.severe_repetition(hyp),
                })
    nll = loss_sum / max(1, tokens)
    result = {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": tokens,
        "route_mass_by_level": (route_sum / max(1, route_batches)).tolist(),
    }
    if generate:
        result.update({
            "exact": exact / max(1, count),
            "nonempty": nonempty / max(1, count),
            "severe_repetition_rate": repeated / max(1, count),
            "token_bleu4": base.bleu4(hypotheses, references),
            "examples": examples,
        })
    return result


def train_arm(
    arm: str, route_mode: str, args, rows, valid_loader, test_loader,
    vocab: int, pad: int, bos: int, eos: int, sp, output: Path,
):
    c01.set_seed(args.model_seed)
    model = load_model(args, vocab, pad).to(args.device)
    encoder_before = tensor_digest(model.encoder)
    trainable = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    batches = data_dose.infinite_batches(
        rows[: args.train_samples], args, pad, args.model_seed + args.train_samples,
    )
    initial = evaluate(model, valid_loader, args, pad, bos, eos, sp, route_mode)
    trace = []
    branch_grad_nonzero = 0
    branch_grad_observations = 0
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
        branch_grad = model.decoder.branch.weight.grad
        if branch_grad is not None:
            branch_grad_observations += 1
            branch_grad_nonzero += int(float(branch_grad.detach().abs().max()) > 0.0)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1

        if step % args.eval_interval == 0 or step == args.fixed_steps:
            valid = evaluate(model, valid_loader, args, pad, bos, eos, sp, route_mode)
            row = {
                "arm": arm, "step": step,
                "train_nll_window": window_loss / max(1, window_steps),
                "valid_nll": valid["nll"],
                "route_mass_by_level": valid["route_mass_by_level"],
                "branch_grad_nonzero_fraction": (
                    branch_grad_nonzero / max(1, branch_grad_observations)
                ),
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            window_loss = 0.0
            window_steps = 0

    final_valid = evaluate(
        model, valid_loader, args, pad, bos, eos, sp, route_mode,
    )
    final = evaluate(
        model, test_loader, args, pad, bos, eos, sp, route_mode, generate=True,
    )
    native = evaluate(model, test_loader, args, pad, bos, eos, sp, "native")
    force_root = evaluate(model, test_loader, args, pad, bos, eos, sp, "force_root")
    force_leaf = evaluate(model, test_loader, args, pad, bos, eos, sp, "force_leaf")
    detail_rows = []
    for depth in range(model.encoder.depths):
        score = evaluate(
            model, test_loader, args, pad, bos, eos, sp, route_mode,
            intervention=f"detail_shuffle_{depth}",
        )
        detail_rows.append({
            "depth": depth,
            "nll": score["nll"],
            "damage_nll": score["nll"] - final["nll"],
        })
    encoder_after = tensor_digest(model.encoder)
    checkpoint_path = output / "checkpoints" / f"decoder_{arm}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        "arm": arm,
        "route_mode": route_mode,
        "decoder_state_dict": {
            key: value.detach().cpu()
            for key, value in model.decoder.state_dict().items()
        },
        "source_checkpoint": args.c04_checkpoint,
    }, checkpoint_path)
    return {
        "arm": arm,
        "route_mode": route_mode,
        "initial_valid": initial,
        "final_valid": final_valid,
        "final_test": final,
        "cross_route": {
            "native": native, "force_root": force_root, "force_leaf": force_leaf,
        },
        "detail_shuffle": detail_rows,
        "max_detail_shuffle_damage_nll": max(row["damage_nll"] for row in detail_rows),
        "branch_grad_nonzero_fraction": (
            branch_grad_nonzero / max(1, branch_grad_observations)
        ),
        "branch_grad_observations": branch_grad_observations,
        "finite_gradients": finite,
        "encoder_digest_before": encoder_before,
        "encoder_digest_after": encoder_after,
        "encoder_unchanged": encoder_before == encoder_after,
        "trace": trace,
        "seconds": time.time() - started,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": c01.file_digest(checkpoint_path),
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
        raise ValueError(f"formal C05 requires {FORMAL_STEPS} updates per arm")

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
    platform_checks = c03.verify_frozen_platform(manifest, baseline, args.smoke)
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

    leaf = arms["leaf_pressure"]
    root = arms["root_control"]
    leaf_gain = leaf["initial_valid"]["nll"] - leaf["final_valid"]["nll"]
    gates = {
        "G1_leaf_nll_gain_at_least_0_30": leaf_gain >= 0.30,
        "G2_leaf_branch_gradient_nonzero": (
            leaf["finite_gradients"]
            and leaf["branch_grad_nonzero_fraction"] > 0.99
        ),
        "G3_detail_shuffle_damage_at_least_0_10": (
            leaf["max_detail_shuffle_damage_nll"] >= 0.10
        ),
        "G4_encoder_unchanged": all(row["encoder_unchanged"] for row in arms.values()),
        "G5_leaf_within_0_10_of_root_control": (
            leaf["final_test"]["nll"] <= root["final_test"]["nll"] + 0.10
        ),
    }
    mechanism_supported = all(gates[key] for key in (
        "G1_leaf_nll_gain_at_least_0_30",
        "G2_leaf_branch_gradient_nonzero",
        "G3_detail_shuffle_damage_at_least_0_10",
        "G4_encoder_unchanged",
    ))
    summary = {
        "experiment_id": "s3_stone1_frozen_encoder_pressure_decoder",
        "claim": "S3-STONE1-FROZEN-PRESSURE-C05",
        "predict": "P-S3-STONE1-FROZEN-PRESSURE-05",
        "status": (
            "forced_recursive_channel_supported_single_seed"
            if mechanism_supported else
            "forced_recursive_channel_not_supported_under_recipe"
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
        "platform_checks": platform_checks,
        "dataset": manifest,
        "source_checkpoint": args.c04_checkpoint,
        "arms": arms,
        "leaf_nll_gain": leaf_gain,
        "leaf_minus_root_final_nll": (
            leaf["final_test"]["nll"] - root["final_test"]["nll"]
        ),
        "gates": gates,
        "boundary": (
            "Forced depth diagnoses a trainable decoder channel; it does not "
            "show spontaneous stopping-depth learning or complete STONE-1."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "leaf_nll_gain": leaf_gain,
        "leaf_minus_root_final_nll": summary["leaf_minus_root_final_nll"],
        "gates": gates,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
