#!/usr/bin/env python3
"""Fine-tune a materialized TreeHeap checkpoint for Chinese dialogue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_c09_replication as c09
import s3_stone1_fixed_root_noise_repair as c08
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base
from treeheap_translate_cli import load_runtime, translate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument(
        "--data",
        default=(
            "/home/nio/datasets/pretrain/Chinese-Train-Datasets/"
            "belle_zh/Belle_open_source_1M.json"
        ),
    )
    parser.add_argument(
        "--product-dir",
        default=(
            "ara/s3-generation/evidence/s3_stone2_product_checkpoint"
        ),
    )
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--valid-samples", type=int, default=1_000)
    parser.add_argument("--test-samples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=72002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def row_digest(source, target):
    return hashlib.sha256(
        (source + "\0" + target).encode("utf-8")
    ).hexdigest()


def load_rows(cli, sp, eos):
    caps = {
        "train": cli.train_samples,
        "valid": cli.valid_samples,
        "test": cli.test_samples,
    }
    splits = {key: [] for key in caps}
    digests = {key: hashlib.sha256() for key in caps}
    with open(cli.data, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            instruction = str(row.get("instruction", "")).strip()
            extra = str(row.get("input", "")).strip()
            target_text = str(row.get("output", "")).strip()
            source_text = instruction + ("\n" + extra if extra else "")
            if not source_text or not target_text:
                continue
            digest = row_digest(source_text, target_text)
            bucket = int(digest[:8], 16) % 1000
            split = "test" if bucket < 10 else "valid" if bucket < 20 else "train"
            if len(splits[split]) >= caps[split]:
                continue
            source = sp.encode(source_text, out_type=int)[:63] + [eos]
            target = sp.encode(target_text, out_type=int)[:63] + [eos]
            if len(source) < 3 or len(target) < 2:
                continue
            splits[split].append((source, target))
            digests[split].update((digest + "\n").encode("ascii"))
            if all(len(splits[key]) >= caps[key] for key in caps):
                break
    if any(len(splits[key]) < caps[key] for key in caps):
        raise RuntimeError({
            key: (len(splits[key]), caps[key]) for key in caps
        })
    manifest = {
        key: {
            "rows": len(splits[key]),
            "sha256": digests[key].hexdigest(),
        }
        for key in caps
    }
    return splits, manifest


def make_loader(rows, batch, pad, workers, shuffle):
    return DataLoader(
        base.ParallelDataset(rows),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=base.collate(pad),
        pin_memory=True,
    )


def main():
    cli = parse_args()
    output = Path(cli.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        (Path(cli.product_dir) / "checkpoints").glob("treeheap_*.pt")
    )
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one product checkpoint, got {checkpoints}")
    checkpoint = checkpoints[0]
    payload, args, tokens, sp, model = load_runtime(
        str(checkpoint), cli.device,
        str(Path(cli.product_dir) / "treeheap_sp.model"),
    )
    args.device = cli.device
    args.batch_size = cli.batch_size
    splits, manifest = load_rows(cli, sp, tokens["eos"])
    loaders = {
        key: make_loader(
            rows, cli.batch_size, tokens["pad"], cli.num_workers,
            key == "train",
        )
        for key, rows in splits.items()
    }
    c01.set_seed(cli.seed)
    initial = c09.evaluate(
        model, loaders["valid"], args, tokens["pad"], tokens["bos"],
        tokens["eos"], sp.get_piece_size(), sp,
    )
    encoder_before = c05.tensor_digest(model.encoder)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cli.encoder_lr},
        {"params": model.decoder.parameters(), "lr": cli.decoder_lr},
    ])
    trace = []
    grad_nonzero = grad_observations = 0
    started = time.time()
    model.train()
    for step, (source, length, target, _) in enumerate(
        loaders["train"], start=1,
    ):
        source = source.to(cli.device)
        length = length.to(cli.device)
        target = target.to(cli.device)
        fixed, visible_length = c08.fixed_source(
            source, length, "eos_tail", args.heap_width, tokens["pad"],
            tokens["eos"], sp.get_piece_size(), args.noise_seed,
        )
        logits, _ = model.teacher(
            fixed, visible_length, target, tokens["bos"],
            route_mode="depth_floor",
        )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=tokens["pad"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        maximum = max(
            (
                float(parameter.grad.detach().abs().max())
                for parameter in model.encoder.parameters()
                if parameter.grad is not None
            ),
            default=0.0,
        )
        grad_observations += 1
        grad_nonzero += int(maximum > 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % cli.eval_interval == 0 or step == math.ceil(
            cli.train_samples / cli.batch_size
        ):
            valid = c09.evaluate(
                model, loaders["valid"], args, tokens["pad"], tokens["bos"],
                tokens["eos"], sp.get_piece_size(), sp,
            )
            event = {
                "step": step,
                "train_nll": float(loss.detach()),
                "valid_nll": valid["nll"],
                "elapsed_sec": time.time() - started,
            }
            trace.append(event)
            print(json.dumps(event), flush=True)

    final_test = c09.evaluate(
        model, loaders["test"], args, tokens["pad"], tokens["bos"],
        tokens["eos"], sp.get_piece_size(), sp, generate=True,
    )
    detail = []
    for depth in range(model.encoder.depths):
        damaged = c09.evaluate(
            model, loaders["test"], args, tokens["pad"], tokens["bos"],
            tokens["eos"], sp.get_piece_size(), sp,
            intervention=f"detail_shuffle_{depth}",
        )
        detail.append({
            "depth": depth,
            "damage_nll": damaged["nll"] - final_test["nll"],
        })
    checkpoint_out = output / "treeheap_dialogue_100k.pt"
    torch.save({
        **payload,
        "task": "zh_dialogue",
        "source_checkpoint": str(checkpoint),
        "model_state_dict": model.state_dict(),
        "student_args": vars(args),
    }, checkpoint_out)
    frozen_prompts = [
        sp.decode(source) for source, _ in splits["test"][:32]
    ]
    before_reload = [
        translate(prompt, args, tokens, sp, model, 64)
        for prompt in frozen_prompts
    ]
    encoder_changed = (
        encoder_before != c05.tensor_digest(model.encoder)
        and grad_nonzero / max(1, grad_observations) > 0.0
    )
    del model
    torch.cuda.empty_cache()
    _, reload_args, reload_tokens, reload_sp, reload_model = load_runtime(
        str(checkpoint_out),
        cli.device,
        str(Path(cli.product_dir) / "treeheap_sp.model"),
    )
    after_reload = [
        translate(
            prompt, reload_args, reload_tokens, reload_sp, reload_model, 64,
        )
        for prompt in frozen_prompts
    ]
    prompts = [
        "你好，请介绍一下你自己。",
        "地球为什么是圆的？",
        "如果今天很疲惫，应该怎样安排工作？",
        "请用简单的话解释什么是递归。",
        "写一句关于夏天晚风的话。",
    ]
    prompt_outputs = [
        {
            "prompt": prompt,
            "response": translate(
                prompt, reload_args, reload_tokens, reload_sp, reload_model, 64,
            ),
        }
        for prompt in prompts
    ]
    gates = {
        "P1_nll_improves_0_20": (
            initial["nll"] - trace[-1]["valid_nll"] >= 0.20
        ),
        "P2_nonempty": final_test["nonempty"] == 1.0,
        "P3_repetition": final_test["severe_repetition_rate"] <= 0.10,
        "P4_detail_causal": max(
            row["damage_nll"] for row in detail
        ) >= 0.10,
        "P5_depth_floor": min(
            final_test["route_mass_by_level"]
        ) >= 0.019,
        "P6_encoder_changed": encoder_changed,
        "P7_reload_exact": before_reload == after_reload,
    }
    summary = {
        "claim": "S3-STONE2-DIALOGUE-C01",
        "status": "supported_pilot" if all(gates.values()) else "open",
        "host": socket.gethostname(),
        "source_checkpoint": str(checkpoint),
        "dataset": manifest,
        "initial_valid": initial,
        "trace": trace,
        "test": final_test,
        "detail_shuffle": detail,
        "prompt_outputs": prompt_outputs,
        "checkpoint": {
            "path": str(checkpoint_out),
            "bytes": checkpoint_out.stat().st_size,
        },
        "gates": gates,
        "boundary": (
            "Single-turn research PoC; not evidence of factuality, safety, "
            "multi-turn memory, consciousness, or commercial licensing."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "test": {
            "nll": final_test["nll"],
            "bleu4": final_test["token_bleu4"],
            "nonempty": final_test["nonempty"],
            "repetition": final_test["severe_repetition_rate"],
        },
        "gates": gates,
        "prompt_outputs": prompt_outputs,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
