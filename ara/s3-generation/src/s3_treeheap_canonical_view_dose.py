#!/usr/bin/env python3
"""Matched additive-dose screen for TreeHeap identity and Butterfly views."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_treeheap_butterfly_bilingual_full as base  # noqa: E402
import s3_treeheap_canonical_view_ratio as c05  # noqa: E402


CLAIM = "S3-TREEHEAP-CANONICAL-DOSE-C06"
ARMS = ("A_base", "S_substitute", "BB_add_butterfly", "BI_add_identity")
EXTRA_RATIO = 0.2


def subset_tensors(source, length, target, batch, seed: int):
    indices = [
        index for index, row in enumerate(batch)
        if c05.canonical_row(int(row[-1]), seed, EXTRA_RATIO)
    ]
    if not indices:
        return None
    index = torch.tensor(indices, device=source.device)
    local_length = length[index]
    local_source = source[index, : int(local_length.max().item())]
    return local_source, local_length, target[index], indices


def mode_loss(model, source, length, target, mode: str, bos: int, pad: int, vocab: int):
    previous = model.encoder.runtime_mode
    try:
        model.encoder.runtime_mode = mode
        logits, _ = model.teacher(source, length, target, bos)
        tokens = int(target.ne(pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / max(1, tokens)
        return loss, tokens
    finally:
        model.encoder.runtime_mode = previous


def update(model, optimizer, loss) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise RuntimeError("non-finite gradient")
    grad_norm = c05.communication_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return grad_norm


def train_arm(checkpoint, arm: str, args, sp, direction_ids, pad, bos, eos, vocab,
              valid_rows, grammar_rows, dreams):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    output = Path(args.evidence_dir) / f"{arm}_seed{args.seed}"
    summary_path = output / "summary.json"
    if summary_path.is_file() and args.resume_completed:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty arm: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = c05.make_model(checkpoint, args, vocab, pad)
    optimizer = c05.make_optimizer(model, checkpoint, args)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    start_line = int(checkpoint.get("next_line", 0) if args.start_line < 0 else args.start_line)
    stop_line = start_line + args.train_lines
    counts = {
        "base_examples": 0, "base_tokens": 0,
        "extra_examples": 0, "extra_tokens": 0,
        "butterfly_examples": 0, "butterfly_tokens": 0,
        "identity_examples": 0, "identity_tokens": 0,
    }
    updates = 0
    weighted_loss = 0.0
    gradient_norms = []
    trace_path = output / "trace.jsonl"
    started = time.time()

    for cursor, block in base.iter_blocks(
        args.data, start_line, args.block_lines, max_lines=stop_line,
    ):
        rows, prepare_counts = base.prepare_train_block(
            block, checkpoint_epoch, args, sp, direction_ids, eos,
        )
        rng = random.Random(args.batch_seed + checkpoint_epoch * 1_000_003 + cursor)
        model.train()
        for batch in base.rows_to_batches(rows, args, rng):
            source, length, target = base.collate(batch, pad, args.device)
            base_tokens = int(target.ne(pad).sum())

            if arm == "S_substitute":
                loss, view_examples, view_tokens = c05.grouped_view_loss(
                    model, source, length, target, batch, EXTRA_RATIO, args.seed,
                    bos, pad, vocab,
                )
                gradient_norms.append(update(model, optimizer, loss))
                updates += 1
                weighted_loss += float(loss.detach()) * base_tokens
                counts["base_examples"] += len(batch)
                counts["base_tokens"] += base_tokens
                for mode in ("butterfly", "identity"):
                    counts[f"{mode}_examples"] += view_examples[mode]
                    counts[f"{mode}_tokens"] += view_tokens[mode]
                continue

            loss, local_tokens = mode_loss(
                model, source, length, target, "butterfly", bos, pad, vocab,
            )
            base_loss = loss
            base_local_tokens = local_tokens
            counts["base_examples"] += len(batch)
            counts["base_tokens"] += local_tokens
            counts["butterfly_examples"] += len(batch)
            counts["butterfly_tokens"] += local_tokens

            if arm not in ("BB_add_butterfly", "BI_add_identity"):
                gradient_norms.append(update(model, optimizer, base_loss))
                updates += 1
                weighted_loss += float(base_loss.detach()) * base_local_tokens
                continue
            subset = subset_tensors(source, length, target, batch, args.seed)
            if subset is None:
                gradient_norms.append(update(model, optimizer, base_loss))
                updates += 1
                weighted_loss += float(base_loss.detach()) * base_local_tokens
                continue
            local_source, local_length, local_target, indices = subset
            extra_mode = "butterfly" if arm == "BB_add_butterfly" else "identity"
            extra_loss, extra_local_tokens = mode_loss(
                model, local_source, local_length, local_target,
                extra_mode, bos, pad, vocab,
            )
            combined_tokens = base_local_tokens + extra_local_tokens
            combined_loss = (
                base_loss * base_local_tokens + extra_loss * extra_local_tokens
            ) / combined_tokens
            gradient_norms.append(update(model, optimizer, combined_loss))
            updates += 1
            weighted_loss += (
                float(base_loss.detach()) * base_local_tokens
                + float(extra_loss.detach()) * extra_local_tokens
            )
            counts["extra_examples"] += len(indices)
            counts["extra_tokens"] += extra_local_tokens
            counts[f"{extra_mode}_examples"] += len(indices)
            counts[f"{extra_mode}_tokens"] += extra_local_tokens

        event = {
            "event": "dose_block", "claim": CLAIM, "arm": arm,
            "cursor": cursor, "updates": updates, "counts": counts.copy(),
            "prepare_counts": prepare_counts, "elapsed_sec": time.time() - started,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    metrics = c05.evaluate_model(
        model, valid_rows, grammar_rows, args, pad, bos, eos, vocab,
    )
    total_tokens = counts["butterfly_tokens"] + counts["identity_tokens"]
    summary = {
        "claim": CLAIM, "arm": arm, "seed": args.seed,
        "start_line": start_line, "stop_line": stop_line,
        "updates": updates, "counts": counts,
        "total_training_tokens": total_tokens,
        "identity_token_ratio": counts["identity_tokens"] / max(1, total_tokens),
        "train_nll": weighted_loss / max(1, total_tokens),
        "communication_grad_norm_mean": sum(gradient_norms) / max(1, len(gradient_norms)),
        "communication_grad_norm_min": min(gradient_norms, default=0.0),
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
        "config": vars(args),
    }
    dream_args = argparse.Namespace(**vars(args))
    dream_args.evidence_dir = str(output)
    selected_dreams = dreams if args.dream_limit <= 0 else dreams[:args.dream_limit]
    summary["dream"] = str(base.render_dreams(
        model, selected_dreams, dream_args, sp, direction_ids,
        pad, bos, eos, vocab, counts["base_examples"], summary,
    ))
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize(args):
    root = Path(args.evidence_dir)
    rows = {}
    for arm in ARMS:
        path = root / f"{arm}_seed{args.seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[arm] = json.loads(path.read_text(encoding="utf-8"))

    a, s, bb, bi = (rows[name] for name in ARMS)
    n_a = a["metrics"]["native"]["mean"]
    n_s = s["metrics"]["native"]["mean"]
    n_bb = bb["metrics"]["native"]["mean"]
    n_bi = bi["metrics"]["native"]["mean"]
    denominator = n_s - n_a
    recovery = (n_s - n_bi) / denominator if denominator > 0 else None
    js_specificity = bb["metrics"]["cross_view_js"] - bi["metrics"]["cross_view_js"]
    native_cost = n_bi - n_bb
    dose_match = (
        s["counts"]["identity_tokens"] == bb["counts"]["extra_tokens"]
        == bi["counts"]["extra_tokens"]
        and a["counts"]["butterfly_tokens"] == bi["counts"]["base_tokens"]
        and bb["counts"]["extra_examples"] == bi["counts"]["extra_examples"]
    )
    structural = (
        bi["metrics"]["source_shuffle_damage"] > 1.5
        and bi["metrics"]["adjacent_damage"] > 0
        and bi["metrics"]["communication_delta_rms"] > 0
        and bi["communication_grad_norm_mean"] > 0
    )
    gates = {
        "P1_recovery_ge_0p50": recovery is not None and recovery >= 0.50,
        "P2_js_specificity_ge_0p05": js_specificity >= 0.05,
        "P3_native_cost_le_0p015": native_cost <= 0.015,
        "P4_structural_source_causality": structural,
        "dose_and_replay_match": dose_match,
    }
    summary = {
        "claim": CLAIM, "seed": args.seed, "arms": rows,
        "derived": {
            "native_nll": {name: row["metrics"]["native"]["mean"] for name, row in rows.items()},
            "cross_view_js": {name: row["metrics"]["cross_view_js"] for name, row in rows.items()},
            "native_dose_recovery": recovery,
            "js_specificity_BB_minus_BI": js_specificity,
            "equal_compute_native_cost_BI_minus_BB": native_cost,
        },
        "gates": gates,
        "screening_status": "screening_positive" if all(gates.values()) else "screening_not_confirmed",
        "boundary": "One seed cannot upgrade C06; confirmation requires a separate preregistration.",
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    base.notify("TreeHeap canonical-view dose C06 finished", path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=(*ARMS, "smoke", "summarize"), required=True)
    p.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt")
    p.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    p.add_argument("--dreams", default="ara/s3-generation/dreams.txt")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_treeheap_canonical_view_dose")
    p.add_argument("--smoke-marker", default="ara/s3-generation/evidence/s3_treeheap_canonical_view_dose/SMOKE_PASS")
    p.add_argument("--require-smoke", action="store_true")
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--start-line", type=int, default=-1)
    p.add_argument("--train-lines", type=int, default=300_000)
    p.add_argument("--block-lines", type=int, default=75_000)
    p.add_argument("--eval-pairs", type=int, default=1_000)
    p.add_argument("--eval-scan", type=int, default=3_000_000)
    p.add_argument("--diagnostic-rows", type=int, default=128)
    p.add_argument("--diagnostic-batch", type=int, default=4)
    p.add_argument("--dream-limit", type=int, default=0)
    p.add_argument("--dream-max-output", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-seed", type=int, default=8104)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reuse-optimizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-completed", action="store_true")
    p.add_argument("--heap-width", type=int, default=256)
    p.add_argument("--max-content", type=int, default=253)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--coupling-scale", type=float, default=0.25)
    p.add_argument("--batch-32", type=int, default=64)
    p.add_argument("--batch-64", type=int, default=32)
    p.add_argument("--batch-128", type=int, default=16)
    p.add_argument("--batch-256", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.arm == "summarize":
        summarize(args)
        return
    marker = Path(args.smoke_marker)
    if args.require_smoke and not marker.is_file():
        raise RuntimeError(f"smoke gate missing: {marker}")
    random.seed(args.batch_seed)
    torch.manual_seed(args.batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.batch_seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    eval_rows = base.read_eval_rows(args, sp, direction_ids, eos)
    dreams = base.parse_dreams(Path(args.dreams))
    grammar_rows = c05.reference_dream_rows(dreams, sp, direction_ids, eos, args)

    arms = ARMS if args.arm == "smoke" else (args.arm,)
    results = []
    for arm in arms:
        results.append(train_arm(
            checkpoint, arm, args, sp, direction_ids, pad, bos, eos, vocab,
            eval_rows["valid"], grammar_rows, dreams,
        ))
    if args.arm == "smoke":
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "claim": CLAIM, "status": "pass", "arms": list(ARMS),
            "train_lines": args.train_lines, "time": time.time(),
        }, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", "arm": args.arm, "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
