#!/usr/bin/env python3
"""Matched continuation-training screen for TreeHeap canonical view ratios."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_treeheap_butterfly_bilingual_full as base  # noqa: E402
from s2_treeheap_butterfly_wmt import ButterflyRecursive  # noqa: E402


CLAIM = "S3-TREEHEAP-CANONICAL-VIEW-C05"


def parse_csv(text: str, cast):
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("empty comma-separated argument")
    return values


def ratio_tag(ratio: float) -> str:
    return f"p{int(round(ratio * 100)):03d}"


def canonical_row(line_no: int, seed: int, ratio: float) -> bool:
    bucket = base.stable_int(f"canonical-view:{seed}:{line_no}") % 1_000_000
    return bucket < int(round(ratio * 1_000_000))


def make_model(checkpoint, args, vocab: int, pad: int):
    config = checkpoint.get("config", {})
    model = ButterflyRecursive(
        vocab,
        int(config.get("dim", args.dim)),
        int(config.get("hidden", args.hidden)),
        int(config.get("heap_width", args.heap_width)),
        pad,
        "butterfly",
        float(config.get("coupling_scale", args.coupling_scale)),
        dynamic_width=True,
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def make_optimizer(model, checkpoint, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.reuse_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    return optimizer


def grouped_view_loss(
    model, source, length, target, batch, ratio: float, seed: int,
    bos: int, pad: int, vocab: int,
):
    canonical = [canonical_row(int(row[-1]), seed, ratio) for row in batch]
    groups = {
        "butterfly": [index for index, flag in enumerate(canonical) if not flag],
        "identity": [index for index, flag in enumerate(canonical) if flag],
    }
    loss_sum = torch.zeros((), device=source.device)
    token_count = 0
    view_examples = {"butterfly": 0, "identity": 0}
    view_tokens = {"butterfly": 0, "identity": 0}
    previous = model.encoder.runtime_mode
    try:
        for mode, indices in groups.items():
            if not indices:
                continue
            index = torch.tensor(indices, device=source.device)
            local_source = source[index]
            local_length = length[index]
            local_target = target[index]
            model.encoder.runtime_mode = mode
            logits, _ = model.teacher(local_source, local_length, local_target, bos)
            local_tokens = int(local_target.ne(pad).sum())
            loss_sum = loss_sum + F.cross_entropy(
                logits.reshape(-1, vocab), local_target.reshape(-1),
                ignore_index=pad, reduction="sum",
            )
            token_count += local_tokens
            view_examples[mode] += len(indices)
            view_tokens[mode] += local_tokens
    finally:
        model.encoder.runtime_mode = previous
    return loss_sum / max(1, token_count), view_examples, view_tokens


def communication_grad_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.encoder.communication.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return float(total.sqrt())


@torch.no_grad()
def communication_delta(model, rows, args, pad: int) -> float:
    batch = rows[: min(args.diagnostic_batch, len(rows))]
    source, length, _ = base.collate(batch, pad, args.device)
    raw, mask = model.encoder.raw_leaf(source, length)
    transformed = model.encoder.communication(raw, mask, "butterfly")
    return float((transformed - raw).float().square().mean().sqrt())


def diagnostic_batches(rows, size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.no_grad()
def cross_view_js(model, rows, args, pad: int, bos: int) -> float:
    model.eval()
    chosen = rows[: min(args.diagnostic_rows, len(rows))]
    total_js = 0.0
    total_tokens = 0
    previous = model.encoder.runtime_mode
    try:
        for batch in diagnostic_batches(chosen, args.diagnostic_batch):
            source, length, target = base.collate(batch, pad, args.device)
            logs = []
            for mode in ("butterfly", "identity"):
                model.encoder.runtime_mode = mode
                logits, _ = model.teacher(source, length, target, bos)
                logs.append(F.log_softmax(logits.float(), dim=-1))
            log_p, log_q = logs
            log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
            js = 0.5 * (
                log_p.exp() * (log_p - log_m)
                + log_q.exp() * (log_q - log_m)
            ).sum(dim=-1)
            mask = target.ne(pad)
            total_js += float(js.masked_select(mask).sum())
            total_tokens += int(mask.sum())
    finally:
        model.encoder.runtime_mode = previous
    return total_js / max(1, total_tokens)


@torch.no_grad()
def source_shuffle_metrics(model, rows, args, pad: int, bos: int, vocab: int):
    model.eval()
    chosen = rows[: min(args.diagnostic_rows, len(rows))]
    native_loss = 0.0
    shuffled_loss = 0.0
    total_tokens = 0
    previous = model.encoder.runtime_mode
    model.encoder.runtime_mode = "butterfly"
    try:
        for batch in diagnostic_batches(chosen, args.diagnostic_batch):
            if len(batch) < 2:
                continue
            source, length, target = base.collate(batch, pad, args.device)
            native_logits, _ = model.teacher(source, length, target, bos)
            shuffled_logits, _ = model.teacher(
                source.roll(1, dims=0), length.roll(1, dims=0), target, bos,
            )
            native_loss += float(F.cross_entropy(
                native_logits.reshape(-1, vocab), target.reshape(-1),
                ignore_index=pad, reduction="sum",
            ))
            shuffled_loss += float(F.cross_entropy(
                shuffled_logits.reshape(-1, vocab), target.reshape(-1),
                ignore_index=pad, reduction="sum",
            ))
            total_tokens += int(target.ne(pad).sum())
    finally:
        model.encoder.runtime_mode = previous
    native_nll = native_loss / max(1, total_tokens)
    shuffled_nll = shuffled_loss / max(1, total_tokens)
    return {
        "matched_native_nll": native_nll,
        "shuffled_nll": shuffled_nll,
        "damage": shuffled_nll - native_nll,
        "tokens": total_tokens,
    }


def reference_dream_rows(dreams, sp, direction_ids, eos: int, args):
    rows = []
    for direction, source, reference in dreams:
        if not reference:
            continue
        pair = (reference, source) if direction == "en2zh" else (source, reference)
        encoded = base.encode_pair(pair, sp, direction, direction_ids, eos)
        if base.eligible(encoded, args.max_content):
            rows.append((*encoded, direction, pair, -1))
    return rows


@torch.no_grad()
def evaluate_model(
    model, valid_rows, grammar_rows, args, pad: int, bos: int, eos: int,
    vocab: int,
):
    native = base.evaluate(model, valid_rows, args, pad, bos, eos, mode="butterfly")
    identity = base.evaluate(model, valid_rows, args, pad, bos, eos, mode="identity")
    adjacent = base.evaluate(model, valid_rows, args, pad, bos, eos, mode="adjacent")
    shuffled = source_shuffle_metrics(model, valid_rows, args, pad, bos, vocab)
    result = {
        "native": native,
        "identity": identity,
        "adjacent": adjacent,
        "identity_damage": identity["mean"] - native["mean"],
        "adjacent_damage": adjacent["mean"] - native["mean"],
        "cross_view_js": cross_view_js(model, valid_rows, args, pad, bos),
        "source_shuffle": shuffled,
        "source_shuffle_damage": shuffled["damage"],
        "communication_delta_rms": communication_delta(model, valid_rows, args, pad),
    }
    if grammar_rows:
        result["grammar_reference_native"] = base.evaluate(
            model, grammar_rows, args, pad, bos, eos, mode="butterfly",
        )
        result["grammar_reference_identity"] = base.evaluate(
            model, grammar_rows, args, pad, bos, eos, mode="identity",
        )
    return result


def train_arm(
    checkpoint, ratio: float, seed: int, args, sp, direction_ids,
    pad: int, bos: int, eos: int, vocab: int, valid_rows, grammar_rows, dreams,
):
    tag = f"{ratio_tag(ratio)}_seed{seed}"
    output = Path(args.evidence_dir) / tag
    output.mkdir(parents=True, exist_ok=True)
    model = make_model(checkpoint, args, vocab, pad)
    optimizer = make_optimizer(model, checkpoint, args)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    start_line = args.start_line
    if start_line < 0:
        start_line = int(checkpoint.get("next_line", 0))
    stop_line = start_line + args.train_lines
    view_examples = {"butterfly": 0, "identity": 0}
    view_tokens = {"butterfly": 0, "identity": 0}
    gradient_norms = []
    train_loss_sum = 0.0
    updates = examples = target_tokens = 0
    started = time.time()

    for cursor, block in base.iter_blocks(
        args.data, start_line, args.block_lines, max_lines=stop_line,
    ):
        rows, counts = base.prepare_train_block(
            block, checkpoint_epoch, args, sp, direction_ids, eos,
        )
        rng = random.Random(args.batch_seed + checkpoint_epoch * 1_000_003 + cursor)
        model.train()
        for batch in base.rows_to_batches(rows, args, rng):
            source, length, target = base.collate(batch, pad, args.device)
            loss, local_examples, local_tokens = grouped_view_loss(
                model, source, length, target, batch, ratio, seed, bos, pad, vocab,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradient in {tag}")
            gradient_norms.append(communication_grad_norm(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach())
            updates += 1
            examples += len(batch)
            target_tokens += int(target.ne(pad).sum())
            for mode in view_examples:
                view_examples[mode] += local_examples[mode]
                view_tokens[mode] += local_tokens[mode]
        print(json.dumps({
            "event": "ratio_block", "arm": tag, "cursor": cursor,
            "updates": updates, "examples": examples,
            "target_tokens": target_tokens, "counts": counts,
        }), flush=True)

    metrics = evaluate_model(
        model, valid_rows, grammar_rows, args, pad, bos, eos, vocab,
    )
    summary = {
        "claim": CLAIM,
        "arm": tag,
        "canonical_probability": ratio,
        "seed": seed,
        "start_line": start_line,
        "stop_line": stop_line,
        "updates": updates,
        "examples": examples,
        "target_tokens": target_tokens,
        "view_examples": view_examples,
        "view_tokens": view_tokens,
        "actual_canonical_example_ratio": view_examples["identity"] / max(1, examples),
        "actual_canonical_token_ratio": view_tokens["identity"] / max(1, target_tokens),
        "train_nll_per_update": train_loss_sum / max(1, updates),
        "communication_grad_norm_mean": sum(gradient_norms) / max(1, len(gradient_norms)),
        "communication_grad_norm_min": min(gradient_norms, default=0.0),
        "elapsed_sec": time.time() - started,
        "metrics": metrics,
    }
    arm_args = argparse.Namespace(**vars(args))
    arm_args.evidence_dir = str(output)
    selected_dreams = dreams if args.dream_limit <= 0 else dreams[: args.dream_limit]
    dream_path = base.render_dreams(
        model, selected_dreams, arm_args, sp, direction_ids,
        pad, bos, eos, sp.get_piece_size(), examples, summary,
    )
    summary["dream"] = str(dream_path)
    if args.save_arm_checkpoints:
        checkpoint_path = output / "checkpoint_final.pt"
        base.atomic_checkpoint(checkpoint_path, {
            "claim": CLAIM,
            "ratio": ratio,
            "seed": seed,
            "state_dict": model.state_dict(),
            "config": vars(args),
            "summary": summary,
        })
        summary["checkpoint"] = str(checkpoint_path)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def screening_decision(results):
    grouped = {}
    for row in results:
        grouped.setdefault(float(row["canonical_probability"]), []).append(row)
    aggregates = {}
    for ratio, rows in grouped.items():
        aggregates[ratio] = {
            "ratio": ratio,
            "seeds": [row["seed"] for row in rows],
            "native_nll": sum(row["metrics"]["native"]["mean"] for row in rows) / len(rows),
            "cross_view_js": sum(row["metrics"]["cross_view_js"] for row in rows) / len(rows),
            "source_shuffle_damage": sum(row["metrics"]["source_shuffle_damage"] for row in rows) / len(rows),
            "communication_grad_norm": sum(row["communication_grad_norm_mean"] for row in rows) / len(rows),
            "communication_delta_rms": sum(row["metrics"]["communication_delta_rms"] for row in rows) / len(rows),
            "identity_damage": sum(row["metrics"]["identity_damage"] for row in rows) / len(rows),
            "adjacent_damage": sum(row["metrics"]["adjacent_damage"] for row in rows) / len(rows),
        }
    baseline = aggregates.get(0.0)
    if baseline is None:
        return {"status": "incomplete", "reason": "missing p=0 continuation control"}
    winner = min(aggregates.values(), key=lambda row: row["native_nll"])
    baseline_nll = baseline["native_nll"]
    winner_nll = winner["native_nll"]
    baseline_js = baseline["cross_view_js"]
    winner_js = winner["cross_view_js"]
    ratio = float(winner["ratio"])
    structural = (
        winner["source_shuffle_damage"] > 0
        and winner["communication_grad_norm"] > 0
        and winner["communication_delta_rms"] > 0
        and max(winner["identity_damage"], winner["adjacent_damage"]) > 0
    )
    return {
        "status": "screening_positive" if (
            ratio in (0.2, 0.4)
            and baseline_nll - winner_nll >= 0.02
            and winner_js <= baseline_js * 0.8
            and structural
        ) else "screening_not_confirmed",
        "winner": ratio_tag(ratio),
        "winner_ratio": ratio,
        "native_nll_gain_vs_p0": baseline_nll - winner_nll,
        "cross_view_js_reduction": 1.0 - winner_js / max(1e-12, baseline_js),
        "structural_gate": structural,
        "aggregates": list(aggregates.values()),
        "note": "single-seed screening cannot upgrade the claim without confirmation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt",
    )
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--dreams", default="ara/s3-generation/dreams.txt")
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_treeheap_canonical_view_ratio",
    )
    parser.add_argument("--ratios", default="0.0,0.2,0.4,0.6")
    parser.add_argument("--seeds", default="9101")
    parser.add_argument("--start-line", type=int, default=-1)
    parser.add_argument("--train-lines", type=int, default=300_000)
    parser.add_argument("--block-lines", type=int, default=50_000)
    parser.add_argument("--eval-pairs", type=int, default=1_000)
    parser.add_argument("--eval-scan", type=int, default=3_000_000)
    parser.add_argument("--diagnostic-rows", type=int, default=128)
    parser.add_argument("--diagnostic-batch", type=int, default=4)
    parser.add_argument("--dream-limit", type=int, default=0)
    parser.add_argument("--dream-max-output", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-seed", type=int, default=8104)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-arm-checkpoints", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--max-content", type=int, default=253)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--coupling-scale", type=float, default=0.25)
    parser.add_argument("--batch-32", type=int, default=64)
    parser.add_argument("--batch-64", type=int, default=32)
    parser.add_argument("--batch-128", type=int, default=16)
    parser.add_argument("--batch-256", type=int, default=8)
    args = parser.parse_args()

    ratios = parse_csv(args.ratios, float)
    seeds = parse_csv(args.seeds, int)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("ratios must be in [0, 1]")
    if 0.0 not in ratios:
        raise ValueError("registered experiment requires the p=0 continuation control")
    if args.train_lines <= 0 or args.block_lines <= 0:
        raise ValueError("train-lines and block-lines must be positive")

    random.seed(args.batch_seed)
    torch.manual_seed(args.batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.batch_seed)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    eval_rows = base.read_eval_rows(args, sp, direction_ids, eos)
    dreams = base.parse_dreams(Path(args.dreams))
    grammar_rows = reference_dream_rows(dreams, sp, direction_ids, eos, args)

    initial_model = make_model(checkpoint, args, vocab, pad)
    initial = evaluate_model(
        initial_model, eval_rows["valid"], grammar_rows,
        args, pad, bos, eos, vocab,
    )
    del initial_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    startup = {
        "claim": CLAIM,
        "checkpoint": str(checkpoint_path),
        "checkpoint_claim": checkpoint.get("claim"),
        "checkpoint_examples": checkpoint.get("global_examples"),
        "checkpoint_tokens": checkpoint.get("global_tokens"),
        "ratios": ratios,
        "seeds": seeds,
        "initial_metrics": initial,
        "config": vars(args),
    }
    (output / "startup.json").write_text(
        json.dumps(startup, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"event": "startup", **startup}, ensure_ascii=False), flush=True)

    results = []
    for seed in seeds:
        for ratio in ratios:
            results.append(train_arm(
                checkpoint, ratio, seed, args, sp, direction_ids,
                pad, bos, eos, vocab, eval_rows["valid"], grammar_rows, dreams,
            ))
    decision = screening_decision(results)
    summary = {
        "claim": CLAIM,
        "initial_metrics": initial,
        "results": results,
        "screening_decision": decision,
        "claim_status": "preregistered / screening only",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if args.notify:
        base.notify("TreeHeap canonical-view ratio screen finished", summary_path)
    print(json.dumps({"event": "finished", "summary": str(summary_path), "decision": decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
