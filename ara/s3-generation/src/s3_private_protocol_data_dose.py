#!/usr/bin/env python3
"""Fixed-update data-dose test for Flat, TreeHeap h1, and Transformer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as adaptive
import s3_private_protocol_battle as battle
import s3_private_protocol_transformer_benchmark as tfbench
import s3_wmt_treeheap_seq2seq as base

Row = Tuple[List[int], List[int]]


def row_key(row: Row) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return tuple(row[0]), tuple(row[1])


def rows_digest(rows: Sequence[Row]) -> str:
    digest = hashlib.sha256()
    for source, target in rows:
        digest.update(struct.pack("<I", len(source)))
        digest.update(struct.pack(f"<{len(source)}I", *source))
        digest.update(struct.pack("<I", len(target)))
        digest.update(struct.pack(f"<{len(target)}I", *target))
    return digest.hexdigest()


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sampling_copy(args, train: int, valid: int, test: int, scan: int, seed: int):
    config = copy.copy(args)
    config.train_samples = train
    config.valid_samples = valid
    config.test_samples = test
    config.max_scan = scan
    config.seed = seed
    return config


def build_nested_data(args, sp, output: Path):
    base_required = args.base_train_samples + args.valid_samples + args.test_samples
    base_args = sampling_copy(
        args, args.base_train_samples, args.valid_samples, args.test_samples,
        args.baseline_max_scan, args.data_seed,
    )
    historical, base_sampling = adaptive.load_rows(base_args, sp)
    if len(historical) != base_required:
        raise RuntimeError("historical split size changed")

    historical_train = historical[: args.base_train_samples]
    valid = historical[
        args.base_train_samples : args.base_train_samples + args.valid_samples
    ]
    test = historical[args.base_train_samples + args.valid_samples :]
    frozen = {row_key(row) for row in valid}
    frozen.update(row_key(row) for row in test)

    # Request a small surplus because duplicates and frozen evaluation rows are
    # removed before constructing the nested train prefixes.
    pool_required = max(args.doses) + base_required
    pool_args = sampling_copy(
        args, pool_required, 0, 0, args.pool_max_scan, args.pool_seed,
    )
    print(json.dumps({"event": "pool_sampling_started", "required": pool_required}), flush=True)
    pool, pool_sampling = adaptive.load_rows(pool_args, sp)

    nested: List[Row] = []
    seen = set()
    removed_eval_overlap = removed_train_duplicate = 0
    for row in historical_train:
        key = row_key(row)
        if key in frozen:
            removed_eval_overlap += 1
            continue
        if key in seen:
            removed_train_duplicate += 1
            continue
        seen.add(key)
        nested.append(row)
    for row in pool:
        if len(nested) >= max(args.doses):
            break
        key = row_key(row)
        if key in frozen or key in seen:
            continue
        seen.add(key)
        nested.append(row)
    if len(nested) < max(args.doses):
        raise RuntimeError(
            f"only {len(nested)} unique leakage-free train rows; need {max(args.doses)}"
        )
    del pool

    split_hashes = {str(dose): rows_digest(nested[:dose]) for dose in args.doses}
    validation_keys = {row_key(row) for row in valid}
    test_keys = {row_key(row) for row in test}
    manifest = {
        "source": {
            "path": args.data,
            "bytes": os.path.getsize(args.data),
            "declared_rows": args.source_rows,
        },
        "tokenizer": {
            "path": args.spm_model,
            "bytes": os.path.getsize(args.spm_model),
            "sha256": file_digest(args.spm_model),
            "vocabulary_pieces": sp.get_piece_size(),
        },
        "language_direction": "en_to_zh",
        "filter": {"min_tokens": args.min_len, "max_tokens": args.max_len},
        "sampling": {
            "historical_30k_contract": base_sampling,
            "large_pool": pool_sampling,
            "data_seed": args.data_seed,
            "pool_seed": args.pool_seed,
            "algorithm": "historical reservoir plus leakage-free deterministic reservoir pool",
        },
        "splits": {
            "train_doses": list(args.doses),
            "train_sha256": split_hashes,
            "validation_rows": len(valid),
            "validation_sha256": rows_digest(valid),
            "test_rows": len(test),
            "test_sha256": rows_digest(test),
        },
        "leakage_control": {
            "train_rows_unique": len(seen),
            "historical_train_eval_overlap_removed": removed_eval_overlap,
            "historical_train_duplicates_removed": removed_train_duplicate,
            "validation_test_content_overlap": len(validation_keys & test_keys),
            "new_train_rows_matching_frozen_eval": 0,
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({"event": "dataset_ready", **manifest["splits"]}), flush=True)
    return nested, valid, test, manifest


def make_loader(rows, args, pad: int, shuffle: bool, generator=None):
    return DataLoader(
        base.ParallelDataset(rows),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=base.collate(pad),
        pin_memory=args.device.startswith("cuda"),
    )


def make_model(name: str, args, vocab: int, pad: int):
    if name in {"flat", "h1"}:
        return battle.make_model(name, vocab, args, pad)
    if name == "transformer":
        return tfbench.make_model(args, vocab, pad, "same_recipe")
    raise ValueError(name)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_batches(rows, args, pad: int, seed: int):
    cycle = 0
    while True:
        generator = torch.Generator().manual_seed(seed + 1009 * cycle)
        loader = make_loader(rows, args, pad, True, generator)
        for batch in loader:
            yield cycle, batch
        cycle += 1


def append_trace(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_arm(
    model_name: str, dose: int, rows: Sequence[Row], valid_loader, test_loader,
    args, vocab: int, pad: int, bos: int, eos: int, sp, output: Path,
):
    seed = args.model_seed
    set_seed(seed)
    model = make_model(model_name, args, vocab, pad).to(args.device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    arm_started = time.time()
    trace_path = output / "trace.jsonl"

    initial = battle.evaluate(model, valid_loader, args, pad, bos, eos, sp)
    best_nll = initial["nll"]
    best_step = 0
    best_state = copy.deepcopy({
        key: value.detach().cpu() for key, value in model.state_dict().items()
    })
    initial_row = {
        "event": "evaluation", "model": model_name, "dose": dose,
        "step": 0, "cycle": 0, "train_nll_window": None,
        "valid_nll": initial["nll"], "elapsed_sec": 0.0,
    }
    append_trace(trace_path, initial_row)
    print(json.dumps(initial_row), flush=True)

    batches = infinite_batches(rows, args, pad, seed + dose)
    window_loss = 0.0
    window_steps = 0
    gradients_finite = True
    last_cycle = 0
    for step in range(1, args.fixed_steps + 1):
        last_cycle, (source, length, target, _) = next(batches)
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        model.train()
        logits, _ = model.teacher(source, length, target, bos)
        loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        window_loss += float(loss.detach())
        window_steps += 1

        evaluate_now = step % args.eval_interval == 0 or step == args.fixed_steps
        if not evaluate_now:
            continue
        valid_score = battle.evaluate(model, valid_loader, args, pad, bos, eos, sp)
        row = {
            "event": "evaluation", "model": model_name, "dose": dose,
            "step": step, "cycle": last_cycle,
            "sample_exposures": step * args.batch_size,
            "train_nll_window": window_loss / max(1, window_steps),
            "valid_nll": valid_score["nll"],
            "elapsed_sec": time.time() - arm_started,
        }
        append_trace(trace_path, row)
        print(json.dumps(row), flush=True)
        window_loss = 0.0
        window_steps = 0
        if math.isfinite(valid_score["nll"]) and valid_score["nll"] < best_nll:
            best_nll = valid_score["nll"]
            best_step = step
            best_state = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })

    model.load_state_dict(best_state)
    test = battle.evaluate(
        model, test_loader, args, pad, bos, eos, sp, generate=True,
    )
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
    )
    result = {
        "model": model_name,
        "dose": dose,
        "seed": seed,
        "parameters": parameters,
        "fixed_steps": args.fixed_steps,
        "sample_exposures": args.fixed_steps * args.batch_size,
        "reuse_factor": args.fixed_steps * args.batch_size / dose,
        "best_step": best_step,
        "best_valid_nll": best_nll,
        "test": test,
        "seconds": time.time() - arm_started,
        "peak_vram_bytes": peak_vram,
        "finite_gradients": gradients_finite,
    }
    del model, best_state
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            result[index] = average
        cursor = end
    return result


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def summarize(results: Sequence[dict], args):
    curves: Dict[str, dict] = {}
    for model in args.models:
        selected = sorted(
            (row for row in results if row["model"] == model),
            key=lambda row: row["dose"],
        )
        doses = [row["dose"] for row in selected]
        nll = [row["test"]["nll"] for row in selected]
        curves[model] = {
            "doses": doses,
            "test_nll": nll,
            "spearman_log_dose_vs_nll": correlation(
                rank([math.log10(value) for value in doses]), rank(nll),
            ),
            "nll_improvement_first_to_last": nll[0] - nll[-1],
            "adjacent_improvements": sum(
                nll[index] < nll[index - 1] for index in range(1, len(nll))
            ),
        }
    primary = curves["h1"]
    gates = {
        "P1_h1_improves_at_least_0_10": primary["nll_improvement_first_to_last"] >= 0.10,
        "P2_h1_spearman_at_most_minus_0_80": primary["spearman_log_dose_vs_nll"] <= -0.80,
        "P3_two_of_three_adjacent_doses_improve": primary["adjacent_improvements"] >= 2,
        "P4_all_gradients_finite": all(row["finite_gradients"] for row in results),
    }
    gain = primary["nll_improvement_first_to_last"]
    if args.smoke:
        status = "smoke_only"
    elif all(gates.values()):
        status = "supported_pilot_single_seed"
    elif gain >= 0.03:
        status = "partial_pilot_single_seed"
    else:
        status = "not_supported_pilot_single_seed"
    return curves, gates, status


def report_markdown(summary: dict, results: Sequence[dict], manifest: dict) -> str:
    cfg = summary["config"]
    lines = [
        "# Controlled Data-Dose Experiment Report",
        "",
        "## Experiment Card",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Experiment ID | `{summary['experiment_id']}` |",
        f"| Claim / Predict | `{summary['claim']}` / `{summary['predict']}` |",
        f"| Status | `{summary['status']}` |",
        f"| Code commit | `{summary['git_commit']}` |",
        f"| Host / device | `{summary['host']}` / `{summary['device_name']}` |",
        f"| Runtime | `{summary['seconds'] / 3600:.2f} h` |",
        "",
        "## Dataset Card",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Source | `{manifest['source']['path']}` |",
        f"| Source bytes / declared rows | {manifest['source']['bytes']:,} / {manifest['source']['declared_rows']:,} |",
        f"| Train doses | {', '.join(f'{x:,}' for x in manifest['splits']['train_doses'])} |",
        f"| Validation | {manifest['splits']['validation_rows']:,} rows, `{manifest['splits']['validation_sha256']}` |",
        f"| Test | {manifest['splits']['test_rows']:,} rows, `{manifest['splits']['test_sha256']}` |",
        f"| Tokenizer | `{manifest['tokenizer']['path']}`, SHA-256 `{manifest['tokenizer']['sha256']}` |",
        f"| Leakage removal | train/eval overlaps removed: {manifest['leakage_control']['historical_train_eval_overlap_removed']}; duplicate train rows removed: {manifest['leakage_control']['historical_train_duplicates_removed']} |",
        "",
        "## Variable Contract",
        "",
        "| Type | Variables |",
        "|---|---|",
        "| Independent | Number of unique training pairs: 30K, 100K, 300K, 1M |",
        f"| Controlled | split hashes, tokenizer, model seed `{cfg['model_seed']}`, `{cfg['fixed_steps']:,}` updates, batch `{cfg['batch_size']}`, AdamW, LR `{cfg['lr']}`, validation cadence `{cfg['eval_interval']}` |",
        "| Dependent | held-out NLL and PPL (lower is better); BLEU4 (higher is better) |",
        "| Known nuisance | single seed; equal steps imply fewer repetitions at larger doses |",
        "",
        "## Results",
        "",
        "| Model | Unique rows | Reuse | Best step | Valid NLL | Test NLL lower-is-better | PPL | BLEU4 | Time | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['dose']:,} | {row['reuse_factor']:.2f}x | "
            f"{row['best_step']:,} | {row['best_valid_nll']:.4f} | "
            f"{row['test']['nll']:.4f} | {row['test']['ppl']:.1f} | "
            f"{row['test']['token_bleu4']:.3f} | {row['seconds'] / 60:.1f} min | "
            f"{row['peak_vram_bytes'] / 2**30:.2f} GiB |"
        )
    lines.extend(["", "## Curves And Decision Gates", "", "```json",
                  json.dumps({"curves": summary["curves"], "gates": summary["gates"]}, indent=2),
                  "```", "", "## Interpretation Boundary", "",
                  summary["boundary"], "",
                  "Per-evaluation values are preserved in `trace.jsonl`; exact split identities are in `dataset_manifest.json`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--models", nargs="+", choices=("h1", "flat", "transformer"), default=["h1", "flat", "transformer"])
    parser.add_argument("--doses", nargs="+", type=int, default=[30000, 100000, 300000, 1000000])
    parser.add_argument("--base-train-samples", type=int, default=30000)
    parser.add_argument("--valid-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--baseline-max-scan", type=int, default=300000)
    parser.add_argument("--pool-max-scan", type=int, default=3000000)
    parser.add_argument("--source-rows", type=int, default=14170275)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71900)
    parser.add_argument("--pool-seed", type=int, default=72003)
    parser.add_argument("--model-seed", type=int, default=71901)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fixed-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--tf-dim", type=int, default=256)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--tf-encoder-layers", type=int, default=2)
    parser.add_argument("--tf-decoder-layers", type=int, default=2)
    parser.add_argument("--tf-feedforward", type=int, default=512)
    parser.add_argument("--tf-dropout", type=float, default=0.0)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.doses = sorted(set(args.doses))
    if "h1" not in args.models:
        raise ValueError("primary h1 model must be included")
    if args.doses[0] != args.base_train_samples:
        raise ValueError("first dose must equal base-train-samples")
    if args.fixed_steps <= 0:
        args.fixed_steps = math.ceil(max(args.doses) / args.batch_size)
    if args.max_len + 1 > args.heap_width:
        raise ValueError("heap width must hold source plus EOS")

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8",
    )
    command = shlex.join(["python3", str(Path(__file__)), *sys.argv[1:]])
    (output / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8",
    )
    trace_path = output / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    rows, valid, test, manifest = build_nested_data(args, sp, output)
    pieces = sp.get_piece_size()
    pad, bos, eos, vocab = pieces, sp.bos_id(), sp.eos_id(), pieces + 1
    valid_loader = make_loader(valid, args, pad, False)
    test_loader = make_loader(test, args, pad, False)

    results = []
    for model_name in args.models:
        for dose in args.doses:
            result = train_arm(
                model_name, dose, rows[:dose], valid_loader, test_loader,
                args, vocab, pad, bos, eos, sp, output,
            )
            results.append(result)
            (output / "runs.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    curves, gates, status = summarize(results, args)
    summary = {
        "experiment_id": "s3_private_protocol_data_dose_full" if not args.smoke else "s3_private_protocol_data_dose_smoke",
        "claim": "S3-PRIVATE-PROTOCOL-DATA-DOSE-C03",
        "predict": "P-S3-PRIVATE-PROTOCOL-DATA-DOSE-03",
        "status": status,
        "host": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "git_commit": git_revision(),
        "seconds": time.time() - started,
        "config": vars(args),
        "curves": curves,
        "gates": gates,
        "boundary": (
            "This fixed-update, single-seed curve tests data diversity under one training recipe. "
            "It does not establish a universal scaling law, dataset optimum, semantic heads, "
            "or TreeHeap superiority. A common gain across models is a corpus effect."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        report_markdown(summary, results, manifest), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
