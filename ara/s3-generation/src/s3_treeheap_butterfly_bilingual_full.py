#!/usr/bin/env python3
"""Resumable full-corpus bilingual training for the Butterfly TreeHeap."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_treeheap_butterfly_wmt import ButterflyRecursive  # noqa: E402


CLAIM = "S3-TREEHEAP-BUTTERFLY-BIDIR-C03"
DIRECTIONS = ("en2zh", "zh2en")


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def partition(line: str) -> str:
    slot = stable_int(line) % 1000
    return "valid" if slot == 0 else "test" if slot == 1 else "train"


def parse_pair(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    zh, en = parts[0].strip(), parts[1].strip()
    return (zh, en) if zh and en else None


def encode_pair(pair, sp, direction: str, direction_ids: dict[str, int], eos: int):
    zh, en = pair
    source, target = (en, zh) if direction == "en2zh" else (zh, en)
    source_ids = [direction_ids[direction], *sp.encode(source, out_type=int), eos]
    target_ids = [*sp.encode(target, out_type=int), eos]
    return source_ids, target_ids


def eligible(encoded, max_content: int) -> bool:
    source, target = encoded
    return 3 <= len(source) <= max_content + 2 and 2 <= len(target) <= max_content + 1


def batch_limit(width: int, args) -> int:
    if width <= 32:
        return args.batch_32
    if width <= 64:
        return args.batch_64
    if width <= 128:
        return args.batch_128
    return args.batch_256


def width_bucket(source, target) -> int:
    needed = max(2, len(source), len(target))
    return min(256, 1 << (needed - 1).bit_length())


def collate(rows, pad: int, device: str):
    source_width = max(len(row[0]) for row in rows)
    target_width = max(len(row[1]) for row in rows)
    source = torch.full((len(rows), source_width), pad, dtype=torch.long)
    target = torch.full((len(rows), target_width), pad, dtype=torch.long)
    length = torch.tensor([len(row[0]) for row in rows], dtype=torch.long)
    for index, (src, tgt, *_rest) in enumerate(rows):
        source[index, : len(src)] = torch.tensor(src)
        target[index, : len(tgt)] = torch.tensor(tgt)
    return source.to(device), length.to(device), target.to(device)


def rows_to_batches(rows, args, rng: random.Random):
    buckets = {32: [], 64: [], 128: [], 256: []}
    rng.shuffle(rows)
    for row in rows:
        width = width_bucket(row[0], row[1])
        bucket = buckets[width]
        bucket.append(row)
        if len(bucket) >= batch_limit(width, args):
            yield bucket[:]
            bucket.clear()
    for width in buckets:
        if buckets[width]:
            yield buckets[width]


def read_eval_rows(args, sp, direction_ids, eos):
    selected = {"valid": [], "test": []}
    with open(args.data, encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            split = partition(line)
            if split not in selected or len(selected[split]) >= args.eval_pairs * 2:
                continue
            pair = parse_pair(line)
            if pair is None:
                continue
            both = []
            for direction in DIRECTIONS:
                encoded = encode_pair(pair, sp, direction, direction_ids, eos)
                if eligible(encoded, args.max_content):
                    both.append((*encoded, direction, pair))
            if len(both) == 2:
                selected[split].extend(both)
            if all(len(rows) >= args.eval_pairs * 2 for rows in selected.values()):
                break
            if args.eval_scan and line_no + 1 >= args.eval_scan:
                break
    for split in selected:
        selected[split] = selected[split][: args.eval_pairs * 2]
    if any(not rows for rows in selected.values()):
        raise RuntimeError(f"failed to build evaluation rows: { {k: len(v) for k,v in selected.items()} }")
    return selected


def iter_blocks(path: str, start_line: int, block_lines: int, max_lines: int = 0):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for _ in range(start_line):
            if not handle.readline():
                return
        cursor = start_line
        while True:
            block = []
            for _ in range(block_lines):
                if max_lines and cursor >= max_lines:
                    break
                line = handle.readline()
                if not line:
                    break
                block.append((cursor, line))
                cursor += 1
            if not block:
                return
            yield cursor, block


def prepare_train_block(block, epoch, args, sp, direction_ids, eos):
    rows = []
    counts = {"raw": len(block), "train_partition": 0, "eligible": 0, "en2zh": 0, "zh2en": 0}
    for line_no, line in block:
        if partition(line) != "train":
            continue
        counts["train_partition"] += 1
        pair = parse_pair(line)
        if pair is None:
            continue
        direction = DIRECTIONS[(stable_int(line) + epoch) & 1]
        encoded = encode_pair(pair, sp, direction, direction_ids, eos)
        if not eligible(encoded, args.max_content):
            continue
        rows.append((*encoded, direction, pair, line_no))
        counts["eligible"] += 1
        counts[direction] += 1
    return rows, counts


def clean(ids: Iterable[int], eos: int, pieces: int):
    output = []
    for token in ids:
        if token == eos:
            break
        if 0 <= token < pieces:
            output.append(token)
    return output


@torch.no_grad()
def evaluate(model, rows, args, pad, bos, eos, mode="butterfly", limit=None):
    model.eval()
    previous = model.encoder.runtime_mode
    model.encoder.runtime_mode = mode
    totals = {direction: [0.0, 0] for direction in DIRECTIONS}
    chosen = rows if limit is None else rows[:limit]
    try:
        rng = random.Random(117)
        for batch in rows_to_batches(chosen[:], args, rng):
            source, length, target = collate(batch, pad, args.device)
            logits, _ = model.teacher(source, length, target, bos)
            for direction in DIRECTIONS:
                indices = [i for i, row in enumerate(batch) if row[2] == direction]
                if not indices:
                    continue
                index = torch.tensor(indices, device=target.device)
                local_logits, local_target = logits[index], target[index]
                loss = F.cross_entropy(
                    local_logits.reshape(-1, local_logits.shape[-1]),
                    local_target.reshape(-1), ignore_index=pad, reduction="sum",
                )
                tokens = int(local_target.ne(pad).sum())
                totals[direction][0] += float(loss)
                totals[direction][1] += tokens
    finally:
        model.encoder.runtime_mode = previous
    result = {direction: totals[direction][0] / max(1, totals[direction][1]) for direction in DIRECTIONS}
    result["mean"] = sum(result.values()) / len(DIRECTIONS)
    return result


def parse_dreams(path: Path):
    dreams = []
    if not path.exists():
        return dreams
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or parts[0] not in DIRECTIONS:
            print(f"dreams.txt:{line_no}: ignored malformed line", flush=True)
            continue
        dreams.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    return dreams


@torch.no_grad()
def render_dreams(model, dreams, args, sp, direction_ids, pad, bos, eos, pieces, step, metrics):
    model.eval()
    lines = [
        f"TreeHeap dreams at step {step}",
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"validation: {json.dumps(metrics, ensure_ascii=False)}",
        "",
    ]
    for number, (direction, source_text, reference) in enumerate(dreams, 1):
        source_ids = [direction_ids[direction], *sp.encode(source_text, out_type=int), eos]
        if len(source_ids) > args.heap_width:
            hypothesis = f"[SKIPPED: {len(source_ids)} pieces exceed heap width {args.heap_width}]"
        else:
            source = torch.tensor([source_ids], device=args.device)
            length = torch.tensor([len(source_ids)], device=args.device)
            generated, route = model.greedy(source, length, bos, eos, args.dream_max_output)
            hypothesis = sp.decode(clean(generated[0].tolist(), eos, pieces))
            route_text = ", ".join(f"{float(v):.4f}" for v in route.cpu())
        lines.extend([
            f"[{number}] {direction}",
            f"SOURCE: {source_text}",
            *( [f"REFERENCE: {reference}"] if reference else [] ),
            f"DREAM: {hypothesis}",
            *( [f"ROUTE: [{route_text}]"] if not hypothesis.startswith("[SKIPPED") else [] ),
            "",
        ])
    text = "\n".join(lines) + "\n"
    dream_dir = Path(args.evidence_dir) / "dreams"
    dream_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dream_dir / f"step-{step:012d}.txt"
    snapshot.write_text(text, encoding="utf-8")
    temporary = dream_dir / "latest.txt.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, dream_dir / "latest.txt")
    return snapshot


def atomic_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def notify(subject: str, report: Path):
    try:
        subprocess.run(["sendme", "-s", subject, "-f", str(report)], check=True, timeout=30)
        return "sent"
    except Exception as error:
        return f"failed: {error}"


def save_state(path, model, optimizer, args, epoch, next_line, examples, tokens, best, started):
    atomic_checkpoint(path, {
        "claim": CLAIM,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
        "epoch": epoch,
        "next_line": next_line,
        "global_examples": examples,
        "global_tokens": tokens,
        "best_valid_mean": best,
        "elapsed_before_resume": time.time() - started,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--dreams", default="ara/s3-generation/dreams.txt")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=8104)
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--max-content", type=int, default=253)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--coupling-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--time-budget-hours", type=float)
    parser.add_argument("--block-lines", type=int)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--wake-blocks", type=int)
    parser.add_argument("--save-blocks", type=int)
    parser.add_argument("--eval-pairs", type=int)
    parser.add_argument("--eval-scan", type=int, default=3_000_000)
    parser.add_argument("--batch-32", type=int, default=64)
    parser.add_argument("--batch-64", type=int, default=32)
    parser.add_argument("--batch-128", type=int, default=16)
    parser.add_argument("--batch-256", type=int, default=8)
    parser.add_argument("--dream-max-output", type=int, default=128)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    defaults = {
        "smoke": dict(epochs=1, time_budget_hours=0.5, block_lines=10_000, wake_blocks=1, save_blocks=1, eval_pairs=128, max_lines=30_000),
        "full": dict(epochs=4, time_budget_hours=96.0, block_lines=50_000, wake_blocks=20, save_blocks=5, eval_pairs=2_000),
    }[args.mode]
    for key, value in defaults.items():
        if getattr(args, key) is None or (key == "max_lines" and args.max_lines == 0 and args.mode == "smoke"):
            setattr(args, key, value)
    if args.heap_width != 256 or args.heap_width & (args.heap_width - 1):
        raise ValueError("this registered run requires a 256-leaf power-of-two TreeHeap")
    if args.max_content + 2 > args.heap_width:
        raise ValueError("direction + content + EOS exceeds heap width")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    model = ButterflyRecursive(
        vocab, args.dim, args.hidden, args.heap_width, pad,
        "butterfly", args.coupling_scale, dynamic_width=True,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    epoch = next_line = global_examples = global_tokens = 0
    best = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch, next_line = int(state["epoch"]), int(state["next_line"])
        global_examples, global_tokens = int(state["global_examples"]), int(state["global_tokens"])
        best = float(state["best_valid_mean"])

    eval_rows = read_eval_rows(args, sp, direction_ids, eos)
    dreams = parse_dreams(Path(args.dreams))
    started = time.time()
    trace_path = output / "trace.jsonl"
    latest_checkpoint = output / "checkpoint_latest.pt"
    block_count = 0

    initial_metrics = evaluate(model, eval_rows["valid"], args, pad, bos, eos, limit=min(256, len(eval_rows["valid"])))
    initial_dream = render_dreams(model, dreams, args, sp, direction_ids, pad, bos, eos, pieces, global_examples, initial_metrics)
    print(json.dumps({"event": "startup", "metrics": initial_metrics, "dream": str(initial_dream)}), flush=True)

    while epoch < args.epochs and time.time() - started < args.time_budget_hours * 3600:
        reached_eof = True
        for cursor, block in iter_blocks(args.data, next_line, args.block_lines, args.max_lines):
            reached_eof = False
            rows, counts = prepare_train_block(block, epoch, args, sp, direction_ids, eos)
            rng = random.Random(args.seed + epoch * 1_000_003 + cursor)
            model.train()
            loss_sum = updates = 0
            for batch in rows_to_batches(rows, args, rng):
                source, length, target = collate(batch, pad, args.device)
                logits, _ = model.teacher(source, length, target, bos)
                loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), ignore_index=pad)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise RuntimeError("non-finite gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach())
                updates += 1
                global_examples += len(batch)
                global_tokens += int(target.ne(pad).sum())
            next_line = cursor
            block_count += 1
            event = {
                "event": "block", "epoch": epoch, "next_line": next_line,
                "global_examples": global_examples, "global_tokens": global_tokens,
                "train_nll": loss_sum / max(1, updates), "counts": counts,
                "elapsed_sec": time.time() - started,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event), flush=True)

            if block_count % args.save_blocks == 0:
                save_state(latest_checkpoint, model, optimizer, args, epoch, next_line, global_examples, global_tokens, best, started)
            if block_count % args.wake_blocks == 0:
                native = evaluate(model, eval_rows["valid"], args, pad, bos, eos)
                identity = evaluate(model, eval_rows["valid"], args, pad, bos, eos, mode="identity", limit=min(512, len(eval_rows["valid"])))
                metrics = {"native": native, "identity": identity, "identity_damage": identity["mean"] - native["mean"]}
                snapshot = render_dreams(model, dreams, args, sp, direction_ids, pad, bos, eos, pieces, global_examples, metrics)
                report = output / "wake_report.json"
                report.write_text(json.dumps({**event, **metrics, "dream": str(snapshot)}, ensure_ascii=False, indent=2), encoding="utf-8")
                if native["mean"] < best:
                    best = native["mean"]
                    save_state(output / "checkpoint_best.pt", model, optimizer, args, epoch, next_line, global_examples, global_tokens, best, started)
                save_state(latest_checkpoint, model, optimizer, args, epoch, next_line, global_examples, global_tokens, best, started)
                status = notify(f"TreeHeap wake-up step {global_examples}", report) if args.notify else "disabled"
                print(json.dumps({"event": "wake", "metrics": metrics, "dream": str(snapshot), "notify": status}), flush=True)
            if time.time() - started >= args.time_budget_hours * 3600:
                break
        if time.time() - started >= args.time_budget_hours * 3600:
            break
        if reached_eof or (args.max_lines and next_line >= args.max_lines):
            epoch += 1
            next_line = 0

    final_native = evaluate(model, eval_rows["test"], args, pad, bos, eos)
    final_identity = evaluate(model, eval_rows["test"], args, pad, bos, eos, mode="identity", limit=min(512, len(eval_rows["test"])))
    final = {
        "claim": CLAIM, "host": socket.gethostname(), "mode": args.mode,
        "epoch": epoch, "next_line": next_line, "global_examples": global_examples,
        "global_tokens": global_tokens, "elapsed_sec": time.time() - started,
        "test_native": final_native, "test_identity": final_identity,
        "identity_damage": final_identity["mean"] - final_native["mean"],
        "best_valid_mean": best, "config": vars(args),
    }
    (output / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    save_state(latest_checkpoint, model, optimizer, args, epoch, next_line, global_examples, global_tokens, best, started)
    final_dream = render_dreams(model, dreams, args, sp, direction_ids, pad, bos, eos, pieces, global_examples, final)
    if args.notify:
        notify(f"TreeHeap {args.mode} training finished", output / "summary.json")
    print(json.dumps({"event": "finished", "summary": final, "dream": str(final_dream)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
