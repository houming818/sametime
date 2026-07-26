#!/usr/bin/env python3
"""One exhaustive BF16 pass over the C10 32K/256-token raw shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_c10_long_smoke as smoke
import s3_stone1_fixed_root_noise_repair as c08
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-dir",
        default="/home/nio/datasets/derived/stone1_c10/raw_32k_256",
    )
    parser.add_argument(
        "--spm-model",
        default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_stone1_c10_raw_full_train",
    )
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--source-width", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=75020)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--valid-every", type=int, default=2000)
    parser.add_argument("--valid-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def load_manifest(block_dir: Path, split: str) -> dict:
    path = block_dir / f"manifest-{split}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("complete_source_pass"):
        raise ValueError(f"{path} is not a complete source pass")
    return value


def shuffled_contract(manifest: dict, seed: int) -> list[int]:
    order = list(range(len(manifest["shards"])))
    random.Random(seed).shuffle(order)
    return order


def row_order(rows: int, seed: int, shard_index: int) -> np.ndarray:
    return np.random.default_rng(seed + shard_index * 1_000_003).permutation(rows)


def batch_tensor(
    block_dir: Path, shard: dict, permutation: np.ndarray, start: int, end: int,
    source_width: int, pad: int, device: str,
):
    values = np.load(block_dir / shard["path"], mmap_mode="r")
    block = np.asarray(values[permutation[start:end]], dtype=np.int64)
    source = torch.from_numpy(block[:, :source_width]).to(device, non_blocking=True)
    target = torch.from_numpy(block[:, source_width:]).to(device, non_blocking=True)
    length = source.ne(pad).sum(1).clamp_min(1)
    return source, length, target


@torch.no_grad()
def evaluate(model, block_dir: Path, manifest: dict, args, pad, bos, eos, pieces,
             intervention: str = "native") -> dict:
    model.eval()
    loss_sum = 0.0
    tokens = 0
    route_sum = None
    batches = 0
    for shard_index, shard in enumerate(manifest["shards"]):
        values = np.load(block_dir / shard["path"], mmap_mode="r")
        for start in range(0, len(values), args.batch):
            block = np.asarray(values[start:start + args.batch], dtype=np.int64)
            source = torch.from_numpy(block[:, :args.source_width]).to(args.device)
            target = torch.from_numpy(block[:, args.source_width:]).to(args.device)
            length = source.ne(pad).sum(1).clamp_min(1)
            fixed, visible = c08.fixed_source(
                source, length, "eos_tail", args.heap_width, pad, eos, pieces,
                args.seed + 97,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                logits, route = model.teacher(
                    fixed, visible, target, bos, intervention=intervention,
                    route_mode="depth_floor",
                )
                loss = F.cross_entropy(
                    logits.flatten(0, 1), target.flatten(), ignore_index=pad,
                    reduction="sum",
                )
            loss_sum += float(loss)
            tokens += int(target.ne(pad).sum())
            current = route.detach().float().cpu()
            route_sum = current if route_sum is None else route_sum + current
            batches += 1
            if batches >= args.valid_batches:
                nll = loss_sum / max(1, tokens)
                return {
                    "nll": nll,
                    "ppl": math.exp(min(20.0, nll)),
                    "tokens": tokens,
                    "route_mass_by_level": (route_sum / batches).tolist(),
                }
    raise ValueError("validation manifest contained no rows")


def main() -> None:
    args = parse_args()
    block_dir = Path(args.block_dir).resolve()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    train_manifest = load_manifest(block_dir, "train")
    valid_manifest = load_manifest(block_dir, "valid")
    manifest_contract = train_manifest["contract"]
    if manifest_contract["width"] != args.source_width + args.target_width:
        raise ValueError("block width and source/target widths disagree")
    spm_path = Path(args.spm_model).resolve()
    if sha256(spm_path) != manifest_contract["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from packed corpus")
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    pieces = sp.get_piece_size()
    pad, bos, eos = pieces, sp.bos_id(), sp.eos_id()
    args.pad, args.bos, args.eos, args.vocab = pad, bos, eos, pieces + 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, floor = smoke.make_model(args, pieces + 1, pad)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    checkpoint_path = evidence / "checkpoint_latest.pt"
    trace_path = evidence / "trace.jsonl"
    shard_order = shuffled_contract(train_manifest, args.seed)
    shard_position = row_position = global_step = processed_rows = processed_tokens = 0
    trace: list[dict] = []
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["shard_order"] != shard_order:
            raise ValueError("checkpoint shard order differs from current contract")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        shard_position = checkpoint["shard_position"]
        row_position = checkpoint["row_position"]
        global_step = checkpoint["global_step"]
        processed_rows = checkpoint["processed_rows"]
        processed_tokens = checkpoint["processed_tokens"]
        trace = checkpoint["trace"]
        print(json.dumps({"resume": True, "global_step": global_step,
                          "shard_position": shard_position,
                          "row_position": row_position}), flush=True)

    initial = evaluate(
        model, block_dir, valid_manifest, args, pad, bos, eos, pieces,
    )
    total_rows = sum(row["rows"] for row in train_manifest["shards"])
    started = time.time()
    interval_started = started
    interval_tokens = 0
    finite = True
    resume_next_position = shard_position
    resume_next_row = row_position
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def save_checkpoint(next_shard_position: int, next_row_position: int) -> None:
        atomic_torch_save(checkpoint_path, {
            "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "shard_order": shard_order,
            "shard_position": next_shard_position,
            "row_position": next_row_position,
            "global_step": global_step,
            "processed_rows": processed_rows,
            "processed_tokens": processed_tokens,
            "trace": trace,
        })

    stopped_early = False
    for position in range(shard_position, len(shard_order)):
        shard_index = shard_order[position]
        shard = train_manifest["shards"][shard_index]
        permutation = row_order(shard["rows"], args.seed, shard_index)
        start_row = row_position if position == shard_position else 0
        for start in range(start_row, shard["rows"], args.batch):
            end = min(start + args.batch, shard["rows"])
            source, length, target = batch_tensor(
                block_dir, shard, permutation, start, end,
                args.source_width, pad, args.device,
            )
            fixed, visible = c08.fixed_source(
                source, length, "eos_tail", args.heap_width, pad, eos, pieces,
                args.seed + 97,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.device.startswith("cuda")):
                logits, _ = model.teacher(
                    fixed, visible, target, bos, route_mode="depth_floor",
                )
                loss = base.ce(logits, target, pad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step_finite = bool(torch.isfinite(loss)) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            finite = finite and step_finite
            if not step_finite:
                save_checkpoint(position, start)
                raise FloatingPointError(f"non-finite loss/gradient at step {global_step + 1}")
            optimizer.step()
            global_step += 1
            rows = end - start
            tokens = int(target.ne(pad).sum())
            processed_rows += rows
            processed_tokens += tokens
            interval_tokens += tokens
            next_position = position
            next_row = end
            if end == shard["rows"]:
                next_position, next_row = position + 1, 0
            resume_next_position, resume_next_row = next_position, next_row
            if global_step == 1 or global_step % args.log_every == 0:
                now = time.time()
                rate = interval_tokens / max(1e-9, now - interval_started)
                eta = (total_rows - processed_rows) * args.target_width / max(rate, 1e-9)
                row = {
                    "step": global_step,
                    "train_nll": float(loss.detach()),
                    "processed_rows": processed_rows,
                    "total_rows": total_rows,
                    "progress": processed_rows / total_rows,
                    "tokens_per_sec": rate,
                    "eta_sec": eta,
                    "shard_position": next_position,
                    "row_position": next_row,
                    "elapsed_sec": now - started,
                }
                trace.append(row)
                print(json.dumps(row), flush=True)
                interval_started, interval_tokens = time.time(), 0
            if global_step % args.valid_every == 0:
                valid = evaluate(
                    model, block_dir, valid_manifest, args, pad, bos, eos, pieces,
                )
                trace[-1 if trace else 0]["valid_nll"] = valid["nll"]
                print(json.dumps({"step": global_step, "valid": valid}), flush=True)
            if global_step % args.checkpoint_every == 0:
                save_checkpoint(next_position, next_row)
            if args.max_steps and global_step >= args.max_steps:
                save_checkpoint(next_position, next_row)
                stopped_early = True
                break
        row_position = 0
        if stopped_early:
            break

    completed = not stopped_early and processed_rows >= total_rows
    final = evaluate(model, block_dir, valid_manifest, args, pad, bos, eos, pieces)
    damages = []
    for depth in range(model.encoder.depths):
        damaged = evaluate(
            model, block_dir, valid_manifest, args, pad, bos, eos, pieces,
            intervention=f"detail_shuffle_{depth}",
        )
        damages.append({"depth": depth, "damage_nll": damaged["nll"] - final["nll"]})
    save_checkpoint(
        len(shard_order) if completed else resume_next_position,
        0 if completed else resume_next_row,
    )
    peak = int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
    summary = {
        "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
        "status": "raw_full_pass_complete" if completed else "stopped_early",
        "host": socket.gethostname(),
        "config": vars(args),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "depth_floor_per_level": floor,
        "manifest": str(block_dir / "manifest-train.json"),
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "processed_tokens": processed_tokens,
        "global_step": global_step,
        "finite": finite,
        "initial": initial,
        "final": final,
        "detail_shuffle": damages,
        "peak_vram_bytes": peak,
        "elapsed_sec_this_run": time.time() - started,
        "trace": trace,
    }
    atomic_json(evidence / "summary.json", summary)
    trace_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "status", "processed_rows", "processed_tokens", "global_step",
        "finite", "peak_vram_bytes", "elapsed_sec_this_run",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
