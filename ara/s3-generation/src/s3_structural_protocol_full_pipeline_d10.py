#!/usr/bin/env python3
"""Resumable clean-corpus pretrain -> bilingual task pipeline for D09 ownership."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_recursive_depth_pressure_protocol_training as d07  # noqa: E402
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_structural_slot_ownership_d08 as d08  # noqa: E402
import s3_structural_slot_ownership_d09_scale as d09  # noqa: E402
import s3_treeheap_butterfly_bilingual_full as wmt  # noqa: E402
import s3_wmt_treeheap_seq2seq as wmt_metrics  # noqa: E402


CLAIM = "S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10"
DEPTHS = d07.DEPTHS
TEXT_SHA256 = "04a90d88b51755561645d0fec962fc8bd5e642d099423348417de9318e22c94e"
PARALLEL_SHA256 = "299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8"
TEXT_ROWS = 2_972_976
PARALLEL_ROWS = 7_304_358


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def file_identity(path: Path, expected_rows: int, expected_sha256: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "expected_rows": expected_rows, "expected_sha256": expected_sha256,
    }


def unfreeze_language(model) -> None:
    for module in (
        model.reconstructor.embedding, model.reconstructor.query,
        model.reconstructor.cell, model.reconstructor.output,
        model.reconstructor.branch, model.reconstructor.depth_embedding,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.freeze_language_backbone = False


def make_model(source_cpu, config, args, warm_path: Path):
    model = d08.OwnershipPressureProtocolModel(
        copy.deepcopy(source_cpu).to(args.device), config.dim, config.hidden,
        args.max_slots, "subheap", args.ownership_seed,
    ).to(args.device)
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    if warm.get("claim") == "S3-STRUCTURAL-SLOT-OWNERSHIP-D09-SCALE":
        d09.load_trainable(model, warm)
        unfreeze_language(model)
    elif warm.get("claim") == CLAIM:
        unfreeze_language(model)
        d09.load_trainable(model, warm)
    else:
        raise RuntimeError(f"unsupported warm checkpoint claim: {warm.get('claim')}")
    return model, warm


def text_partition(identifier: str) -> str:
    slot = stable_int(identifier) % 1000
    return "valid" if slot == 0 else "test" if slot == 1 else "train"


def text_example(payload: dict, sp, seed: int):
    text = str(payload.get("text", ""))
    identifier = str(payload.get("id", ""))
    ids = sp.encode(text, out_type=int)
    widths = [32, 16, 8]
    available = [width for width in widths if len(ids) >= width + 32]
    if not available:
        return None
    width = available[stable_int(f"d10-width:{seed}:{identifier}") % len(available)]
    span = width + 32
    start = stable_int(f"d10-start:{seed}:{identifier}") % (len(ids) - span + 1)
    source = ids[start:start + width]
    target = ids[start + width:start + span]
    return (source, target, "zh_cont", (sp.decode(source), sp.decode(target)), identifier)


def collect_text_eval(path: Path, sp, seed: int, wanted: int):
    selected = {"valid": [], "test": []}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            payload = json.loads(line)
            split = text_partition(str(payload.get("id", "")))
            if split not in selected or len(selected[split]) >= wanted:
                continue
            row = text_example(payload, sp, seed)
            if row is not None:
                selected[split].append(row)
            if all(len(rows) >= wanted for rows in selected.values()):
                break
    if min(map(len, selected.values())) < wanted:
        raise RuntimeError(f"insufficient text eval rows: { {k: len(v) for k,v in selected.items()} }")
    return selected["valid"], selected["test"]


def iter_text_batches(path: Path, sp, seed: int, batch_size: int, start_line: int, max_lines: int):
    batch = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no < start_line:
                continue
            if max_lines and line_no >= max_lines:
                break
            payload = json.loads(line)
            if text_partition(str(payload.get("id", ""))) != "train":
                continue
            row = text_example(payload, sp, seed)
            if row is None:
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                yield line_no + 1, batch
                batch = []
        if batch:
            yield min(line_no + 1, max_lines or line_no + 1), batch


def truncated_pair(pair, sp, direction: str, direction_ids: dict, eos: int):
    zh, en = pair
    source_text, target_text = (en, zh) if direction == "en2zh" else (zh, en)
    source_full = sp.encode(source_text, out_type=int)
    target_full = sp.encode(target_text, out_type=int)
    source = [direction_ids[direction], *source_full[:32], eos]
    target = [*target_full[:63], eos]
    return source, target, len(source_full) > 32, len(target_full) > 63


def collect_wmt_eval(path: Path, sp, direction_ids: dict, eos: int, wanted: int):
    selected = {"valid": [], "test": []}
    pairs = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            split = wmt.partition(line)
            if split not in selected or len(selected[split]) >= wanted * 2:
                continue
            pair = wmt.parse_pair(line)
            if pair is None:
                continue
            pairs.add(pair)
            for direction in wmt.DIRECTIONS:
                source, target, _, _ = truncated_pair(pair, sp, direction, direction_ids, eos)
                selected[split].append((*[source, target], direction, pair, line_no))
            if all(len(rows) >= wanted * 2 for rows in selected.values()):
                break
    if min(map(len, selected.values())) < wanted * 2:
        raise RuntimeError(f"insufficient WMT eval rows: { {k: len(v) for k,v in selected.items()} }")
    return selected["valid"][:wanted * 2], selected["test"][:wanted * 2], pairs


def iter_parallel_batches(
    path: Path, sp, direction_ids: dict, eos: int, batch_size: int,
    start_line: int, max_lines: int, excluded: set,
):
    batch, counts = [], {"rows": 0, "excluded": 0, "source_truncated": 0, "target_truncated": 0}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            if line_no < start_line:
                continue
            if max_lines and line_no >= max_lines:
                break
            pair = wmt.parse_pair(line)
            if pair is None:
                continue
            counts["rows"] += 1
            if pair in excluded:
                counts["excluded"] += 1
                continue
            direction = wmt.DIRECTIONS[stable_int(f"d10-dir:{line_no}:{line}") & 1]
            source, target, source_cut, target_cut = truncated_pair(
                pair, sp, direction, direction_ids, eos,
            )
            counts["source_truncated"] += int(source_cut)
            counts["target_truncated"] += int(target_cut)
            batch.append((source, target, direction, pair, line_no))
            if len(batch) >= batch_size:
                yield line_no + 1, batch, dict(counts)
                batch = []
        if batch:
            yield min(line_no + 1, max_lines or line_no + 1), batch, dict(counts)


@torch.no_grad()
def generation_summary(model, rows, args, sp, pad, bos, eos, pieces):
    per_depth = {}
    for depth in DEPTHS:
        hypotheses, references, examples = [], [], []
        adjacent_equal = adjacent_total = 0
        for row in rows[:args.generation_examples]:
            source, lengths, target = c10.collate_rows([row], pad, args.device)
            generated, route, budgets, _, _ = model.greedy(
                source, lengths, bos, eos, args.max_generation, depth,
            )
            hypothesis = wmt.clean(generated[0].tolist(), eos, pieces)
            reference = wmt.clean(target[0].tolist(), eos, pieces)
            hypotheses.append(hypothesis)
            references.append(reference)
            adjacent_equal += sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
            adjacent_total += max(0, len(hypothesis) - 1)
            if len(examples) < 12:
                examples.append({
                    "kind": row[2], "source": row[3][1] if row[2] == "en2zh" else row[3][0],
                    "reference": sp.decode(reference), "generation": sp.decode(hypothesis),
                    "budget": int(budgets[0]), "route": [float(x) for x in route.detach().cpu()],
                })
        per_depth[str(depth)] = {
            "token_bleu4": wmt_metrics.bleu4(hypotheses, references),
            "nonempty_rate": sum(bool(row) for row in hypotheses) / max(1, len(hypotheses)),
            "adjacent_repetition_rate": adjacent_equal / max(1, adjacent_total),
            "examples": examples,
        }
    bleu = sorted(row["token_bleu4"] for row in per_depth.values())
    return {
        "per_depth": per_depth, "bleu4_median": bleu[len(bleu) // 2],
        "nonempty_min": min(row["nonempty_rate"] for row in per_depth.values()),
        "repetition_max": max(row["adjacent_repetition_rate"] for row in per_depth.values()),
    }


def valid_summary(model, rows, args, pad, bos):
    per_depth = {
        str(depth): d08.evaluate(model, rows, pad, bos, args.device, args.eval_batch, depth)
        for depth in DEPTHS
    }
    return {"per_depth": per_depth, "mean_nll": sum(x["nll"] for x in per_depth.values()) / len(per_depth)}


def causal_summary(model, rows, args, pad, bos):
    result = {}
    for depth in DEPTHS:
        native = d08.evaluate(model, rows, pad, bos, args.device, args.eval_batch, depth)
        shuffle = d08.evaluate(model, rows, pad, bos, args.device, args.eval_batch, depth, "shuffle")
        zero = d08.evaluate(model, rows, pad, bos, args.device, args.eval_batch, depth, "zero")
        result[str(depth)] = {
            "native": native, "shuffle": shuffle, "zero": zero,
            "shuffle_delta": shuffle["nll"] - native["nll"],
            "zero_delta": zero["nll"] - native["nll"],
        }
    return result


def save_checkpoint(path: Path, model, optimizer, stage: str, step: int, cursor: int, payload: dict, progress: dict):
    state = d08.trainable_state(model)
    state_hash = c10.state_sha256(state)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "claim": CLAIM, "stage": stage, "step": step, "cursor": cursor,
        "trainable_state_dict": state, "trainable_state_sha256": state_hash,
        "optimizer_state_dict": optimizer.state_dict(), "run": payload, "progress": progress,
    }, temporary)
    os.replace(temporary, path)
    return state_hash


def run_stage(args, model, warm, config, sp, pad, bos, eos, pieces, valid_rows, test_rows, iterator_factory):
    output = Path(args.evidence_dir) / args.stage
    output.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = output / "checkpoint_latest.pt", output / "checkpoint_best.pt"
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    source_before = c10.state_sha256(model.frozen_source.state_dict())
    language_before = c10.state_sha256(d07.language_backbone_state(model))
    initial_valid = valid_summary(model, valid_rows, args, pad, bos)
    initial_generation = generation_summary(model, test_rows, args, sp, pad, bos, eos, pieces)
    start_step = start_cursor = stale = 0
    best_nll, best_step = initial_valid["mean_nll"], 0
    if args.resume and latest_path.exists():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        if latest.get("claim") != CLAIM or latest.get("stage") != args.stage:
            raise RuntimeError("resume contract mismatch")
        d09.load_trainable(model, latest)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        start_step, start_cursor = int(latest["step"]), int(latest["cursor"])
        initial_valid = latest["run"]["initial_valid"]
        initial_generation = latest["run"]["initial_generation"]
        best_nll = float(latest["progress"]["best_nll"])
        best_step = int(latest["progress"]["best_step"])
        stale = int(latest["progress"]["stale_wakes"])
    contract = {
        "claim": CLAIM, "stage": args.stage, "config": vars(args),
        "source_sha256": source_before, "warm_sha256": warm["trainable_state_sha256"],
        "initial_valid": initial_valid, "initial_generation": initial_generation,
    }
    if not best_path.exists() or start_step == 0:
        save_checkpoint(best_path, model, optimizer, args.stage, start_step, start_cursor, contract, {
            "best_nll": best_nll, "best_step": best_step, "stale_wakes": stale,
        })
    trace = output / "trace.jsonl"
    started, processed_tokens, processed_examples = time.time(), 0, 0
    last_cursor, step = start_cursor, start_step
    stage_counts = {}
    stream_completed = False
    for item in iterator_factory(start_cursor):
        if args.stage == "task":
            cursor, batch, stage_counts = item
        else:
            cursor, batch = item
        step += 1
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        model.train()
        source, lengths, target = c10.collate_rows(batch, pad, args.device)
        logits, route, budgets, slots, entropy = model.teacher(source, lengths, target, bos, depth)
        tokens = int(target.ne(pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pad, reduction="sum",
        ) / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not d07.finite_trainable_gradients(model):
            raise RuntimeError(f"non-finite gradient at {args.stage} step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        processed_tokens += tokens
        processed_examples += len(batch)
        last_cursor = cursor
        if step == start_step + 1 or step % args.log_every == 0:
            event = {
                "event": "train", "stage": args.stage, "step": step, "cursor": cursor,
                "depth": depth, "loss": float(loss.detach()), "grad_norm": grad_norm,
                "tokens_since_resume": processed_tokens, "examples_since_resume": processed_examples,
                "slot_variance": float(slots.detach().var()),
                "route": [float(x) for x in route.detach().cpu()],
                "slot_route": model.compressor.last_route_statistics,
                "entropy": [float(x) for x in entropy.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(trace, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        at_end = args.max_lines and cursor >= args.max_lines
        if step % args.wake_every == 0 or at_end:
            valid = valid_summary(model, valid_rows, args, pad, bos)
            generation = generation_summary(model, test_rows, args, sp, pad, bos, eos, pieces)
            improved = valid["mean_nll"] <= best_nll - args.min_delta
            if improved:
                best_nll, best_step, stale = valid["mean_nll"], step, 0
                best_hash = save_checkpoint(best_path, model, optimizer, args.stage, step, cursor, contract, {
                    "best_nll": best_nll, "best_step": best_step, "stale_wakes": stale,
                })
            else:
                stale += 1
                best_hash = torch.load(best_path, map_location="cpu", weights_only=False)["trainable_state_sha256"]
            latest_hash = save_checkpoint(latest_path, model, optimizer, args.stage, step, cursor, contract, {
                "best_nll": best_nll, "best_step": best_step, "stale_wakes": stale,
            })
            wake = {
                "event": "wake", "stage": args.stage, "step": step, "cursor": cursor,
                "valid": valid, "generation": generation, "improved": improved,
                "best_nll": best_nll, "best_step": best_step, "stale_wakes": stale,
                "latest_sha256": latest_hash, "best_sha256": best_hash,
                "stage_counts": stage_counts, "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "wakes.jsonl", wake)
            write_json(output / "wake_latest.json", wake)
            print(json.dumps(wake, ensure_ascii=False), flush=True)
            if args.stop_on_plateau and step >= args.min_steps and stale >= args.patience:
                break
        if args.max_steps and step >= args.max_steps:
            break
    else:
        stream_completed = True
    if stream_completed:
        last_cursor = args.max_lines

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    d09.load_trainable(model, best)
    final_valid = valid_summary(model, valid_rows, args, pad, bos)
    final_causal = causal_summary(model, test_rows, args, pad, bos)
    final_generation = generation_summary(model, test_rows, args, sp, pad, bos, eos, pieces)
    source_after = c10.state_sha256(model.frozen_source.state_dict())
    language_after = c10.state_sha256(d07.language_backbone_state(model))
    reloaded, _ = make_model(copy.deepcopy(model.frozen_source).cpu(), config, args, best_path)
    reference = valid_summary(model, valid_rows[:64], args, pad, bos)["mean_nll"]
    reload_nll = valid_summary(reloaded, valid_rows[:64], args, pad, bos)["mean_nll"]
    causal_depths = sum(
        row["shuffle_delta"] >= 0.10 and row["zero_delta"] >= 0.10
        for row in final_causal.values()
    )
    structure = all(
        row["native"]["route"]["owner_leaf_coverage"] == 1.0
        and row["native"]["route"]["argmax_coverage"] >= 0.999
        and row["native"]["route"]["route_pair_overlap"] < 0.05
        for row in final_causal.values()
    )
    full_lines = args.max_lines == args.expected_rows and last_cursor >= args.expected_rows
    gates = {
        "data_cursor_complete": full_lines if args.mode == "full" else last_cursor >= args.max_lines,
        "learning": initial_valid["mean_nll"] - final_valid["mean_nll"] >= (0.10 if args.mode == "full" else -1e9),
        "input_causality": causal_depths >= 2 if args.mode == "full" else True,
        "structure": structure,
        "generation": (
            final_generation["nonempty_min"] == 1.0
            and final_generation["repetition_max"] <= (0.10 if args.stage == "task" else 0.25)
            and (args.stage != "task" or args.mode != "full" or
                 final_generation["bleu4_median"] - initial_generation["bleu4_median"] >= 0.50)
        ),
        "source_frozen": source_before == source_after,
        "language_updated": language_before != language_after,
        "reload": abs(reference - reload_nll) < 1e-9 and d09.trainable_hash(reloaded) == best["trainable_state_sha256"],
    }
    summary = {
        "claim": CLAIM, "stage": args.stage, "mode": args.mode,
        "decision": "stage_supported" if all(gates.values()) else "stage_not_supported",
        "gates": gates, "host": socket.gethostname(), "best_step": best_step,
        "cursor": last_cursor, "processed_examples_since_resume": processed_examples,
        "processed_tokens_since_resume": processed_tokens, "stage_counts": stage_counts,
        "initial_valid": initial_valid, "best_valid": final_valid,
        "initial_generation": initial_generation, "best_generation": final_generation,
        "best_causal": final_causal,
        "hashes": {"source_before": source_before, "source_after": source_after,
                   "language_before": language_before, "language_after": language_after,
                   "best_trainable": best["trainable_state_sha256"]},
        "reload_nll_delta": abs(reference - reload_nll), "config": vars(args),
        "seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", "stage": args.stage,
                      "decision": summary["decision"], "gates": gates}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pretrain", "task"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--text-data", default="/home/nio/datasets/nio/releases/NioText-ZH-Integrity-2985K-v1/data.jsonl")
    parser.add_argument("--parallel-data", default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv")
    parser.add_argument("--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11001)
    parser.add_argument("--ownership-seed", type=int, default=11002)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--generation-examples", type=int, default=64)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wake-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-steps", type=int, default=30000)
    parser.add_argument("--stop-on-plateau", action="store_true")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # The frozen C12 source loader retains the historical WMT argument name.
    args.wmt_data = args.eval_wmt_data
    args.expected_rows = TEXT_ROWS if args.stage == "pretrain" else PARALLEL_ROWS
    if args.max_lines <= 0:
        args.max_lines = args.expected_rows if args.mode == "full" else 4096
    if args.mode == "smoke":
        args.eval_rows = min(args.eval_rows, 128)
        args.generation_examples = min(args.generation_examples, 16)
        args.wake_every = min(args.wake_every, 50)
        args.log_every = min(args.log_every, 20)
        args.min_steps = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    text_path, parallel_path = Path(args.text_data), Path(args.parallel_data)
    identities = {
        "text": file_identity(text_path, TEXT_ROWS, TEXT_SHA256),
        "parallel": file_identity(parallel_path, PARALLEL_ROWS, PARALLEL_SHA256),
    }
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad, vocab = pieces, pieces + 3
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    source_cpu, _, config, source_hash, parent_hash = d03.load_model(
        Path(args.source_checkpoint), args, sp, pad, vocab,
    )
    model, warm = make_model(source_cpu, config, args, Path(args.warm_start))
    text_valid, text_test = collect_text_eval(text_path, sp, args.seed, args.eval_rows)
    wmt_valid, wmt_test, excluded = collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows,
    )
    write_json(output / f"contract_{args.stage}.json", {
        "claim": CLAIM, "stage": args.stage, "mode": args.mode,
        "data": identities, "source_sha256": source_hash, "parent_sha256": parent_hash,
        "warm_claim": warm.get("claim"), "warm_sha256": warm["trainable_state_sha256"],
        "tokenizer": c10.sampled_file_fingerprint(Path(args.spm_model)), "config": vars(args),
    })
    if args.stage == "pretrain":
        iterator = lambda cursor: iter_text_batches(
            text_path, sp, args.seed, args.batch_size, cursor, args.max_lines,
        )
        valid_rows, test_rows = text_valid, text_test
    else:
        iterator = lambda cursor: iter_parallel_batches(
            parallel_path, sp, direction_ids, eos, args.batch_size,
            cursor, args.max_lines, excluded,
        )
        valid_rows, test_rows = wmt_valid, wmt_test
    run_stage(
        args, model, warm, config, sp, pad, bos, eos, pieces,
        valid_rows, test_rows, iterator,
    )


if __name__ == "__main__":
    main()
