#!/usr/bin/env python3
"""Teacher-free 128-token / 256-leaf TreeHeap C10 smoke."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_full_corpus_repair_seq2seq as corpus
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_fixed_root_noise_repair as c08
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--data-root", default="/home/nio/datasets")
    parser.add_argument(
        "--spm-model",
        default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument("--source-width", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=128)
    parser.add_argument("--heap-width", type=int, default=256)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--valid-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=75010)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def make_model(args, vocab: int, pad: int):
    model = c01.StoneTreeHeap(
        vocab, args.dim, args.hidden, args.heap_width, pad,
        "learned_structural", 1,
    )
    visible_depths = model.encoder.depths
    floor = 0.12 / visible_depths
    model.decoder = c06.FloorPressureDecoder(
        vocab, args.dim, args.hidden, visible_depths, floor,
    )
    return model, floor


def fixed_source(source, length, args, pad, eos, pieces):
    return c08.fixed_source(
        source, length, "eos_tail", args.heap_width, pad, eos, pieces,
        args.seed + 97,
    )


@torch.no_grad()
def evaluate(model, loader, args, pad, bos, eos, pieces, batches, intervention="native"):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    route_sum = None
    for _ in range(batches):
        source, length, target, _ = next(loader)
        source = source.to(args.device)
        length = length.to(args.device)
        target = target.to(args.device)
        fixed, visible = fixed_source(source, length, args, pad, eos, pieces)
        logits, route = model.teacher(
            fixed, visible, target, bos,
            route_mode="depth_floor", intervention=intervention,
        )
        loss_sum += float(F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), ignore_index=pad,
            reduction="sum",
        ))
        tokens += int(target.ne(pad).sum())
        route = route.detach().float().cpu()
        route_sum = route if route_sum is None else route_sum + route
    nll = loss_sum / max(1, tokens)
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "tokens": tokens,
        "route_mass_by_level": (route_sum / batches).tolist(),
    }


def main():
    args = parse_args()
    if args.heap_width != 2 * args.source_width:
        raise ValueError("C10 smoke preserves the half-text/half-EOS heap contract")
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pad = sp.get_piece_size()
    vocab = pad + 1
    bos, eos, pieces = sp.bos_id(), sp.eos_id(), sp.get_piece_size()
    args.pad, args.vocab, args.bos, args.eos = pad, vocab, bos, eos
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, floor = make_model(args, vocab, pad)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = args.amp and args.device.startswith("cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    train = iter(corpus.make_loader(args, "train", args.seed, args.batch))
    valid_seed = args.seed + 9000
    initial = evaluate(
        model, iter(corpus.make_loader(args, "valid", valid_seed, args.batch)),
        args, pad, bos, eos, pieces, args.valid_batches,
    )
    digest_before = c01.file_digest(Path(args.spm_model))
    counters = Counter()
    trace = []
    finite = True
    started = time.time()
    interval_started = started
    interval_tokens = 0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        source, length, target, source_ids = next(train)
        source = source.to(args.device)
        length = length.to(args.device)
        target = target.to(args.device)
        for source_id in source_ids.tolist():
            counters[corpus.SOURCE_NAMES[source_id]] += 1
        fixed, visible = fixed_source(source, length, args, pad, eos, pieces)
        model.train()
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_enabled,
        ):
            logits, _ = model.teacher(
                fixed, visible, target, bos, route_mode="depth_floor",
            )
            loss = base.ce(logits, target, pad)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_finite = bool(torch.isfinite(loss)) and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        finite = finite and step_finite
        scaler.step(optimizer)
        scaler.update()
        interval_tokens += int(target.ne(pad).sum())
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            now = time.time()
            valid = evaluate(
                model,
                iter(corpus.make_loader(args, "valid", valid_seed, args.batch)),
                args, pad, bos, eos, pieces, args.valid_batches,
            )
            row = {
                "step": step,
                "train_nll": float(loss.detach()),
                "valid_nll": valid["nll"],
                "tokens_per_sec": interval_tokens / max(1e-9, now - interval_started),
                "elapsed_sec": now - started,
                "route_mass_by_level": valid["route_mass_by_level"],
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
            interval_started = time.time()
            interval_tokens = 0

    final_loader = lambda: iter(corpus.make_loader(args, "valid", valid_seed, args.batch))
    final = evaluate(model, final_loader(), args, pad, bos, eos, pieces, args.valid_batches * 2)
    damages = []
    for depth in range(model.encoder.depths):
        damaged = evaluate(
            model, final_loader(), args, pad, bos, eos, pieces,
            args.valid_batches, intervention=f"detail_shuffle_{depth}",
        )
        damages.append({"depth": depth, "damage_nll": damaged["nll"] - final["nll"]})
    checkpoint = output / "checkpoint_latest.pt"
    torch.save({
        "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "step": args.steps,
        "trace": trace,
    }, checkpoint)
    peak = int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
    gates = {
        "finite": finite,
        "learned": initial["nll"] - final["nll"] >= 0.05,
        "all_depths_visible": len(final["route_mass_by_level"]) == 8,
        "depth_floor": min(final["route_mass_by_level"]) >= floor - 1e-4,
        "structural_damage": max(row["damage_nll"] for row in damages) >= 0.01,
        "vram_under_20gib": peak < 20 * 1024**3,
    }
    summary = {
        "claim": "S3-STONE1-FULL-CORPUS-LONG-C10",
        "status": "smoke_pass" if all(gates.values()) else "smoke_failed",
        "host": socket.gethostname(),
        "config": vars(args),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "depth_floor_per_level": floor,
        "initial": initial,
        "final": final,
        "detail_shuffle": damages,
        "source_counts": dict(counters),
        "peak_vram_bytes": peak,
        "tokenizer_sha256": digest_before,
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size},
        "trace": trace,
        "gates": gates,
        "elapsed_sec": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"], "initial_nll": initial["nll"],
        "final_nll": final["nll"], "peak_vram_bytes": peak, "gates": gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
