#!/usr/bin/env python3
"""Resume the paired D11 arms to a complete second corpus pass."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
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
import s3_structural_protocol_capacity_ladder_d11 as d11  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402


CLAIM = "S3-EPOCH-REPEAT-SCALING-E01"
DEPTHS = d07.DEPTHS


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def optimizer_to(optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def direction_for(line_no: int, line: str, flip: int) -> str:
    bit = (d10.stable_int(f"d10-dir:{line_no}:{line}") & 1) ^ (flip & 1)
    return d10.wmt.DIRECTIONS[bit]


def iter_sequential_batches(
    path: Path,
    sp,
    direction_ids: dict,
    eos: int,
    batch_size: int,
    start_line: int,
    start_byte: int,
    end_line: int,
    excluded: set,
    direction_flip: int = 0,
):
    """Yield resumable batches while preserving the original D10 line contract."""
    batch = []
    counts = {
        "rows": 0,
        "examples": 0,
        "excluded": 0,
        "invalid": 0,
        "source_truncated": 0,
        "target_truncated": 0,
    }
    with path.open("rb") as handle:
        line_no = 0
        if start_byte:
            handle.seek(start_byte)
            line_no = start_line
        else:
            while line_no < start_line:
                if not handle.readline():
                    raise RuntimeError(f"EOF while seeking start line {start_line}")
                line_no += 1
        while line_no < end_line:
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace")
            current = line_no
            line_no += 1
            counts["rows"] += 1
            pair = d10.wmt.parse_pair(line)
            if pair is None:
                counts["invalid"] += 1
                continue
            if pair in excluded:
                counts["excluded"] += 1
                continue
            direction = direction_for(current, line, direction_flip)
            source, target, source_cut, target_cut = d10.truncated_pair(
                pair, sp, direction, direction_ids, eos
            )
            counts["source_truncated"] += int(source_cut)
            counts["target_truncated"] += int(target_cut)
            counts["examples"] += 1
            batch.append((source, target, direction, pair, current))
            if len(batch) >= batch_size:
                yield line_no, handle.tell(), batch, dict(counts)
                batch = []
        if batch:
            yield line_no, handle.tell(), batch, dict(counts)


def checkpoint_payload(
    model,
    optimizer,
    run: dict,
    step: int,
    cursor: int,
    byte_offset: int,
    processed_tokens: int,
    processed_examples: int,
    best_nll: float,
    best_step: int,
) -> dict:
    state = d11.trainable_state(model)
    return {
        "claim": CLAIM,
        "arm": run["arm"],
        "step": step,
        "cursor": cursor,
        "byte_offset": byte_offset,
        "corpus_pass": run["corpus_pass"],
        "direction_flip": run["direction_flip"],
        "processed_tokens": processed_tokens,
        "processed_examples": processed_examples,
        "best_nll": best_nll,
        "best_step": best_step,
        "trainable_state_dict": state,
        "trainable_state_sha256": c10.state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "run": run,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_resume(path: Path, arm: str, expected_cursor: int | None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("arm") != arm:
        raise RuntimeError(f"checkpoint arm {payload.get('arm')} != {arm}")
    if expected_cursor is not None and int(payload["cursor"]) != expected_cursor:
        raise RuntimeError(
            f"checkpoint cursor {payload['cursor']} != expected {expected_cursor}"
        )
    if c10.state_sha256(payload["trainable_state_dict"]) != payload["trainable_state_sha256"]:
        raise RuntimeError("resume checkpoint trainable-state hash mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=("treeheap-63m", "treeheap-106m"), required=True)
    parser.add_argument("--extra-dim", type=int, required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--warm-start", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument(
        "--parallel-data",
        default="/home/nio/datasets/nio/releases/NioClean-ZHEN-S098-7M-v2/pairs.tsv",
    )
    parser.add_argument(
        "--eval-wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv"
    )
    parser.add_argument(
        "--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11101)
    parser.add_argument("--ownership-seed", type=int, default=11102)
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--generation-examples", type=int, default=64)
    parser.add_argument("--max-generation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-segment-steps", type=int, default=10000)
    parser.add_argument("--target-cursor", type=int, default=d10.PARALLEL_ROWS)
    parser.add_argument("--expected-start-cursor", type=int)
    parser.add_argument("--corpus-pass", type=int, default=2)
    parser.add_argument("--direction-flip", type=int, choices=(0, 1), default=0)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()
    if (args.arm == "treeheap-63m") != (args.extra_dim == 0):
        raise ValueError("treeheap-63m requires extra_dim=0; treeheap-106m requires extra_dim>0")
    if args.mode == "smoke":
        args.max_segment_steps = min(args.max_segment_steps, 20)
        args.eval_rows = min(args.eval_rows, 32)
        args.generation_examples = min(args.generation_examples, 8)
        args.eval_batch = min(args.eval_batch, 8)
        args.log_every = min(args.log_every, 5)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.wmt_data = args.eval_wmt_data
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    args.pad, args.vocab = pieces, pieces + 3
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    source_cpu, _, config, source_hash, _ = d03.load_model(
        Path(args.source_checkpoint), args, sp, args.pad, args.vocab
    )
    resume = load_resume(
        Path(args.resume_checkpoint), args.arm, args.expected_start_cursor
    )
    model, warm = d11.make_model(
        source_cpu, config, args, Path(args.warm_start), resume
    )
    initial_model_hash = d11.trainable_hash(model)
    if initial_model_hash != resume["trainable_state_sha256"]:
        raise RuntimeError("model differs from resume checkpoint after construction")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr
    )
    optimizer.load_state_dict(resume["optimizer_state_dict"])
    optimizer_to(optimizer, args.device)
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    valid_rows, test_rows, excluded = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.eval_rows
    )
    start_step = int(resume["step"])
    start_cursor = int(resume["cursor"])
    start_byte = int(resume.get("byte_offset", 0))
    processed_tokens = int(resume.get("processed_tokens", 9_536_085))
    processed_examples = int(resume.get("processed_examples", start_cursor))
    best_nll = float(resume.get("best_nll", math.inf))
    best_step = int(resume.get("best_step", start_step))
    initial_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    if not math.isfinite(initial_valid["mean_nll"]):
        raise RuntimeError("non-finite initial NLL")
    if not math.isfinite(best_nll):
        best_nll = initial_valid["mean_nll"]
    run = {
        "claim": CLAIM,
        "mode": args.mode,
        "arm": args.arm,
        "corpus_pass": args.corpus_pass,
        "direction_flip": args.direction_flip,
        "source_sha256": source_hash,
        "warm_start_sha256": warm["trainable_state_sha256"],
        "parallel_sha256": d10.PARALLEL_SHA256,
        "resume_claim": resume.get("claim"),
        "resume_trainable_sha256": resume["trainable_state_sha256"],
        "parameters": d11.parameter_summary(model),
        "config": vars(args),
        "start": {
            "step": start_step,
            "cursor": start_cursor,
            "byte_offset": start_byte,
            "nll": initial_valid["mean_nll"],
        },
    }
    if not (output / "contract.json").exists():
        write_json(output / "contract.json", run)

    started = time.time()
    step = start_step
    cursor = start_cursor
    byte_offset = start_byte
    segment_examples = 0
    segment_tokens = 0
    finite = True
    stage_counts = {}
    iterator = iter_sequential_batches(
        Path(args.parallel_data), sp, direction_ids, eos, args.batch_size,
        start_cursor, start_byte, args.target_cursor, excluded, args.direction_flip,
    )
    for cursor, byte_offset, batch, stage_counts in iterator:
        step += 1
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        model.train()
        source, lengths, target = c10.collate_rows(batch, args.pad, args.device)
        logits, route, _, slots, entropy = model.teacher(source, lengths, target, bos, depth)
        tokens = int(target.ne(args.pad).sum())
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=args.pad, reduction="sum",
        ) / max(1, tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and d07.finite_trainable_gradients(model)
        if not finite or not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"non-finite state at step {step}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        ))
        optimizer.step()
        segment_examples += len(batch)
        segment_tokens += tokens
        processed_examples += len(batch)
        processed_tokens += tokens
        if step == start_step + 1 or step % args.log_every == 0:
            event = {
                "event": "train", "arm": args.arm, "step": step,
                "cursor": cursor, "depth": depth, "loss": float(loss.detach()),
                "grad_norm": grad_norm, "processed_tokens": processed_tokens,
                "processed_examples": processed_examples,
                "slot_variance": float(slots.detach().var()),
                "extra_logit_gain": float(torch.tanh(model.extra_logit_gain).detach()),
                "base_route": model.compressor.last_route_statistics,
                "extra_route": model.last_extra_route_statistics,
                "entropy": [float(value) for value in entropy.detach().cpu()],
                "elapsed_seconds": time.time() - started,
            }
            append_jsonl(output / "trace.jsonl", event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if step - start_step >= args.max_segment_steps:
            break

    if cursor <= start_cursor or step <= start_step:
        raise RuntimeError("data cursor or optimizer step did not advance")
    completed = cursor >= args.target_cursor
    final_valid = d10.valid_summary(model, valid_rows, args, args.pad, bos)
    generation = d10.generation_summary(
        model, test_rows, args, sp, args.pad, bos, eos, pieces
    )
    if final_valid["mean_nll"] < best_nll:
        best_nll = final_valid["mean_nll"]
        best_step = step
        improved = True
    else:
        improved = False
    source_after = c10.state_sha256(model.frozen_source.state_dict())
    payload = checkpoint_payload(
        model, optimizer, run, step, cursor, byte_offset,
        processed_tokens, processed_examples, best_nll, best_step,
    )
    latest = output / "checkpoint_latest.pt"
    save_checkpoint(latest, payload)
    if improved:
        shutil.copy2(latest, output / "checkpoint_best.pt")
    reference_reload_nll = d10.valid_summary(
        model, valid_rows[:32], args, args.pad, bos
    )["mean_nll"]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_model, _ = d11.make_model(
        source_cpu, config, args, Path(args.warm_start), payload
    )
    reloaded_nll = d10.valid_summary(
        reload_model, valid_rows[:32], args, args.pad, bos
    )["mean_nll"]
    reload_delta = abs(reference_reload_nll - reloaded_nll)
    final_hash = payload["trainable_state_sha256"]
    gates = {
        "finite": finite and math.isfinite(final_valid["mean_nll"]),
        "source_frozen": source_hash == source_after,
        "model_updated": initial_model_hash != final_hash,
        "optimizer_loaded": bool(resume.get("optimizer_state_dict")),
        "step_advanced": step > start_step,
        "cursor_advanced": cursor > start_cursor,
        "cursor_bounded": cursor <= args.target_cursor,
        "reload": reload_delta < 1e-9,
    }
    wake = {
        "event": "segment", "claim": CLAIM, "arm": args.arm,
        "corpus_pass": args.corpus_pass, "start_step": start_step,
        "step": step, "start_cursor": start_cursor, "cursor": cursor,
        "target_cursor": args.target_cursor, "completed": completed,
        "segment_examples": segment_examples, "segment_tokens": segment_tokens,
        "processed_examples": processed_examples, "processed_tokens": processed_tokens,
        "initial_valid": initial_valid, "valid": final_valid,
        "generation": generation, "best_nll": best_nll, "best_step": best_step,
        "extra_logit_gain": float(torch.tanh(reload_model.extra_logit_gain).detach()),
        "checkpoint_sha256": final_hash, "reload_nll_delta": reload_delta,
        "stage_counts": stage_counts, "gates": gates,
        "seconds": time.time() - started, "host": socket.gethostname(),
    }
    append_jsonl(output / "wakes.jsonl", wake)
    write_json(output / "wake_latest.json", wake)
    write_json(output / "summary.json", wake)
    receipt = output / "segments" / f"step_{step:07d}_cursor_{cursor:08d}.json"
    write_json(receipt, wake)
    print(json.dumps({
        "event": "segment_complete", "arm": args.arm, "step": step,
        "cursor": cursor, "completed": completed, "gates": gates,
    }, ensure_ascii=False), flush=True)
    if not all(gates.values()):
        raise SystemExit(5)


if __name__ == "__main__":
    main()

