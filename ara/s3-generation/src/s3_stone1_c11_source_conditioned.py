#!/usr/bin/env python3
"""C11 variable-context continuation with an explicit source-dependence gate."""
from __future__ import annotations

import argparse
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
import s3_stone1_c10_long_smoke as c10
import s3_stone1_fixed_root_noise_repair as c08
import s3_wmt_treeheap_seq2seq as base


LENGTHS = (16, 32, 64, 128, 256)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--block-dir", default="/home/nio/datasets/derived/stone1_c10/raw_32k_256")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    p.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_stone1_c10_raw_full_train/checkpoint_latest.pt")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_stone1_c11_source_conditioned")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--gate-step", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--valid-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dependence-weight", type=float, default=0.20)
    p.add_argument("--dependence-margin", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=75031)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class AdjacentBlocks:
    def __init__(self, root: Path, split: str, seed: int):
        manifest = json.loads((root / f"manifest-{split}.json").read_text(encoding="utf-8"))
        self.root = root
        self.shards = [row for row in manifest["shards"] if row["rows"] >= 2]
        self.arrays = [np.load(root / row["path"], mmap_mode="r") for row in self.shards]
        self.rng = np.random.default_rng(seed)

    def batch(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        sources, targets = [], []
        for _ in range(count):
            shard_index = int(self.rng.integers(len(self.shards)))
            values = self.arrays[shard_index]
            index = int(self.rng.integers(len(values) - 1))
            sources.append(np.asarray(values[index], dtype=np.int64))
            targets.append(np.asarray(values[index + 1, :128], dtype=np.int64))
        return np.stack(sources), np.stack(targets)


def variable_source(raw: torch.Tensor, lengths: torch.Tensor, pad: int) -> torch.Tensor:
    out = torch.full_like(raw, pad)
    for row, length in enumerate(lengths.tolist()):
        out[row, :length] = raw[row, -length:]
    return out


def make_batch(stream: AdjacentBlocks, batch: int, pad: int, device: str,
               rng: random.Random, fixed_length: int | None = None):
    source_np, target_np = stream.batch(batch)
    raw = torch.from_numpy(source_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    lengths = torch.tensor(
        [fixed_length or rng.choice(LENGTHS) for _ in range(batch)],
        dtype=torch.long, device=device,
    )
    source = variable_source(raw, lengths, pad)
    return source, lengths, target


def token_nll(logits: torch.Tensor, target: torch.Tensor, pad: int) -> torch.Tensor:
    losses = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=pad, reduction="none")
    mask = target.ne(pad)
    return (losses * mask).sum(1) / mask.sum(1).clamp_min(1)


@torch.no_grad()
def evaluate(model, stream, args, pad, bos, eos, rng, fixed_length: int) -> dict:
    model.eval()
    native_sum = wrong_sum = empty_sum = tokens = 0.0
    js_sum = 0.0
    for _ in range(args.valid_batches):
        source, length, target = make_batch(
            stream, args.batch, pad, args.device, rng, fixed_length,
        )
        wrong = source.roll(1, 0)
        wrong_length = length.roll(1, 0)
        empty = torch.full_like(source, pad)
        empty[:, 0] = eos  # visible EOS marker; no corpus content
        empty_length = torch.ones_like(length)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            native_logits, route = model.teacher(source, length, target, bos, route_mode="depth_floor")
            wrong_logits, _ = model.teacher(wrong, wrong_length, target, bos, route_mode="depth_floor")
            empty_logits, _ = model.teacher(empty, empty_length, target, bos, route_mode="depth_floor")
        count = int(target.ne(pad).sum())
        native_sum += float(F.cross_entropy(native_logits.flatten(0, 1), target.flatten(), ignore_index=pad, reduction="sum"))
        wrong_sum += float(F.cross_entropy(wrong_logits.flatten(0, 1), target.flatten(), ignore_index=pad, reduction="sum"))
        empty_sum += float(F.cross_entropy(empty_logits.flatten(0, 1), target.flatten(), ignore_index=pad, reduction="sum"))
        tokens += count
        p = native_logits[:, 0].float().softmax(-1)
        q = wrong_logits[:, 0].float().softmax(-1)
        m = (p + q) * 0.5
        js_sum += float((0.5 * ((p * (p.clamp_min(1e-9).log() - m.clamp_min(1e-9).log())).sum(-1)
                              + (q * (q.clamp_min(1e-9).log() - m.clamp_min(1e-9).log())).sum(-1))).mean())
    native = native_sum / tokens
    return {
        "native_nll": native,
        "ppl": math.exp(min(20.0, native)),
        "shuffle_nll": wrong_sum / tokens,
        "shuffle_damage": wrong_sum / tokens - native,
        "empty_nll": empty_sum / tokens,
        "empty_damage": empty_sum / tokens - native,
        "first_step_js": js_sum / args.valid_batches,
        "route_mass_by_level": route.detach().float().cpu().tolist(),
        "tokens": int(tokens),
    }


@torch.no_grad()
def samples(model, stream, args, sp, pad, bos, eos, rng) -> list[dict]:
    model.eval()
    source, length, _ = make_batch(stream, 8, pad, args.device, rng, 128)
    predicted, _ = model.greedy(source, length, bos, eos, 48, route_mode="depth_floor")
    rows = []
    for src, n, output in zip(source, length, predicted):
        source_ids = src[:int(n)].detach().cpu().tolist()
        output_ids = base.clean(output.detach().cpu().tolist(), eos, pad)
        rows.append({"source": sp.decode(source_ids), "output": sp.decode(output_ids)})
    return rows


def save_checkpoint(path: Path, model, optimizer, args, step, trace) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({
        "claim": "S3-STONE1-SOURCE-CONDITIONED-C11",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args), "step": step, "trace": trace,
    }, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = args_parser()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, pad, bos, eos = sp.get_piece_size(), sp.get_piece_size(), sp.bos_id(), sp.eos_id()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old = argparse.Namespace(**checkpoint["config"])
    old.heap_width = 256
    args.dim = old.dim
    args.hidden = old.hidden
    args.heap_width = old.heap_width
    args.source_width = 256
    args.target_width = 128
    model, floor = c10.make_model(old, pieces + 1, pad)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train = AdjacentBlocks(Path(args.block_dir), "train", args.seed)
    valid = AdjacentBlocks(Path(args.block_dir), "valid", args.seed + 9000)
    rng = random.Random(args.seed)
    valid_rng = random.Random(args.seed + 8000)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    trace = []
    started = time.time()
    initial = {str(n): evaluate(model, valid, args, pad, bos, eos, valid_rng, n) for n in LENGTHS}
    print(json.dumps({"initial": initial}, ensure_ascii=False), flush=True)
    stopped = False
    for step in range(1, args.steps + 1):
        source, length, target = make_batch(train, args.batch, pad, args.device, rng)
        wrong = source.roll(1, 0)
        wrong_length = length.roll(1, 0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            native_logits, _ = model.teacher(source, length, target, bos, route_mode="depth_floor")
            wrong_logits, _ = model.teacher(wrong, wrong_length, target, bos, route_mode="depth_floor")
            native = token_nll(native_logits, target, pad)
            wrong_loss = token_nll(wrong_logits, target, pad)
            dependence = F.relu(args.dependence_margin + native - wrong_loss).mean()
            loss = native.mean() + args.dependence_weight * dependence
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            metrics = {str(n): evaluate(model, valid, args, pad, bos, eos, valid_rng, n) for n in LENGTHS}
            row = {"step": step, "train_nll": float(native.mean().detach()),
                   "dependence_loss": float(dependence.detach()), "by_length": metrics,
                   "elapsed_sec": time.time() - started}
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            save_checkpoint(evidence / "checkpoint_latest.pt", model, optimizer, args, step, trace)
            if step >= args.gate_step and min(metrics[str(n)]["shuffle_damage"] for n in (64, 128, 256)) < 0.05:
                stopped = True
                print(json.dumps({"stop": "source_dependence_gate_failed", "step": step}), flush=True)
                break
    final = {str(n): evaluate(model, valid, args, pad, bos, eos, valid_rng, n) for n in LENGTHS}
    generated = samples(model, valid, args, sp, pad, bos, eos, valid_rng)
    unique = len({row["output"].strip() for row in generated}) / max(1, len(generated))
    gates = {
        "source_causal_64_256": min(final[str(n)]["shuffle_damage"] for n in (64, 128, 256)) >= 0.05,
        "empty_worse_64_256": min(final[str(n)]["empty_damage"] for n in (64, 128, 256)) >= 0.05,
        "first_step_source_sensitive": min(final[str(n)]["first_step_js"] for n in (64, 128, 256)) > 1e-5,
        "free_output_unique_half": unique >= 0.5,
    }
    summary = {
        "claim": "S3-STONE1-SOURCE-CONDITIONED-C11",
        "status": "completed" if not stopped else "stopped_by_gate",
        "host": socket.gethostname(), "config": vars(args), "initial": initial,
        "final": final, "samples": generated, "unique_output_fraction": unique,
        "gates": gates, "trace": trace, "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()), "depth_floor": floor,
    }
    atomic_json(evidence / "summary.json", summary)
    (evidence / "trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace), encoding="utf-8",
    )
    (evidence / "README.md").write_text(
        "# C11 evidence\n\nSee `summary.json` and `trace.jsonl`. The run may stop at the registered source-dependence gate.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "gates": gates,
                      "unique_output_fraction": unique}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
