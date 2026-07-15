#!/usr/bin/env python3
"""Learn a native TreeHeap encoder/decoder protocol from raw tokens.

There is no pretrained embedding, route supervision, state target, leaf bypass,
or encoder-decoder attention. Surface echo cross-entropy must train the shared
WRITE/FOLD/DETAIL/UNFOLD/READ operators end to end.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def binary_ste(logits: torch.Tensor) -> torch.Tensor:
    """Hard {-1,+1} forward code with tanh straight-through gradients."""
    soft = torch.tanh(logits)
    hard = torch.where(soft >= 0, torch.ones_like(soft), -torch.ones_like(soft))
    return hard.detach() - soft.detach() + soft


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("treeheap_codec_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import data helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeTreeHeapCodec(nn.Module):
    def __init__(self, vocab: int, dim: int, detail_width: int, hidden: int):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.detail_width = detail_width

        # WRITE: learned leaf state from a raw token id.
        self.write = nn.Embedding(vocab, dim)

        # FOLD and DETAIL: shared analysis convolutions at every internal node.
        self.fold = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.detail = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, detail_width),
        )

        # UNFOLD: shared synthesis convolution, recursively reused top-down.
        self.unfold = nn.Sequential(
            nn.Linear(dim + detail_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim),
        )

        # READ: a pointwise shared surface kernel followed by tied token weights.
        self.read = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.output_bias = nn.Parameter(torch.zeros(vocab))

    def encode(self, tokens: torch.Tensor):
        node = self.write(tokens)
        details = []
        while node.shape[1] > 1:
            if node.shape[1] % 2:
                raise ValueError("native codec currently requires a power-of-two length")
            left, right = node[:, 0::2], node[:, 1::2]
            common = (left + right) * math.sqrt(0.5)
            difference = (left - right) * math.sqrt(0.5)
            parent = self.fold(common)
            detail = binary_ste(self.detail(difference))
            details.append(detail)
            node = parent
        return binary_ste(node[:, 0]), details

    def decode(self, root: torch.Tensor, details: List[torch.Tensor], intervention: str = "normal"):
        if intervention == "root_zero":
            root = torch.zeros_like(root)
        current = root[:, None]
        for detail in reversed(details):
            if intervention == "detail_zero":
                detail = torch.zeros_like(detail)
            elif intervention == "detail_shift" and detail.shape[1] > 1:
                detail = detail.roll(1, dims=1)
            elif intervention == "detail_wrong_sample" and detail.shape[0] > 1:
                detail = detail.roll(1, dims=0)
            children = self.unfold(torch.cat((current, detail), dim=-1))
            left, right = children.chunk(2, dim=-1)
            current = torch.stack((left, right), dim=2).flatten(1, 2)
        return current

    def logits(self, leaves: torch.Tensor) -> torch.Tensor:
        state = self.read(leaves)
        return F.linear(state, self.write.weight, self.output_bias)

    def forward(self, tokens: torch.Tensor, intervention: str = "normal"):
        root, details = self.encode(tokens)
        leaves = self.decode(root, details, intervention)
        return self.logits(leaves), root, details, leaves


def batch_stream(base, block_dir: Path, split: str, batch: int, seed: int, max_blocks: int):
    manifest = base.manifest(block_dir, split)
    yield from base.iter_batches(block_dir, manifest, batch, seed, max_blocks)


@torch.no_grad()
def evaluate(model: NativeTreeHeapCodec, base, args, device: torch.device, seed: int, length: int) -> dict:
    model.eval()
    modes = ("normal", "detail_shift", "detail_zero", "detail_wrong_sample", "root_zero")
    totals = {mode: {"loss": 0.0, "correct": 0, "sequence": 0} for mode in modes}
    tokens_seen = 0
    sequences_seen = 0
    examples = []
    for tokens, _ in batch_stream(base, Path(args.block_dir), "valid", args.eval_batch, seed + 9000 + length, args.max_valid_blocks):
        tokens = tokens[:, :length].to(device)
        for mode in modes:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _, _, _ = model(tokens, mode)
                loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten(), reduction="sum")
            pred = logits.argmax(-1)
            exact = pred.eq(tokens)
            totals[mode]["loss"] += float(loss)
            totals[mode]["correct"] += int(exact.sum())
            totals[mode]["sequence"] += int(exact.all(-1).sum())
            if mode == "normal" and len(examples) < 8:
                for gold, output in zip(tokens[:2].cpu().tolist(), pred[:2].cpu().tolist()):
                    examples.append({"gold_ids": gold, "pred_ids": output, "exact": gold == output})
        tokens_seen += tokens.numel()
        sequences_seen += tokens.shape[0]
    metrics = {}
    for mode, row in totals.items():
        metrics[mode] = {
            "nll": row["loss"] / max(1, tokens_seen),
            "token_top1": row["correct"] / max(1, tokens_seen),
            "sequence_exact": row["sequence"] / max(1, sequences_seen),
        }
    return {"length": length, "metrics": metrics, "examples": examples[:8]}


def gradient_audit(model: NativeTreeHeapCodec, tokens: torch.Tensor) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    logits, _, _, _ = model(tokens)
    loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
    loss.backward()

    def norm(parameters) -> float:
        values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return float(torch.stack(values).norm()) if values else 0.0

    result = {
        "loss": float(loss.detach()),
        "write_grad_norm": norm(model.write.parameters()),
        "fold_grad_norm": norm(model.fold.parameters()),
        "detail_grad_norm": norm(model.detail.parameters()),
        "unfold_grad_norm": norm(model.unfold.parameters()),
        "read_grad_norm": norm(model.read.parameters()),
    }
    model.zero_grad(set_to_none=True)
    return result


def run_seed(base, args, device: torch.device, vocab: int, seed: int, out: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = NativeTreeHeapCodec(vocab, args.dim, args.detail_width, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trace_path = out / f"trace_seed_{seed}.jsonl"
    trace_path.unlink(missing_ok=True)
    started = time.time()
    step = 0
    last_tokens = None
    for tokens, _ in batch_stream(base, Path(args.block_dir), "train", args.batch, seed, args.max_train_blocks):
        tokens = tokens.to(device, non_blocking=True)
        last_tokens = tokens
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _, _, _ = model(tokens)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "blocks": min(step * args.batch, args.max_train_blocks),
                "train_nll": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "elapsed_sec": time.time() - started,
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
            }
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"seed": seed, **row}), flush=True)

    if last_tokens is None:
        raise RuntimeError("training stream produced no batches")
    audit = gradient_audit(model, last_tokens[: min(8, last_tokens.shape[0])])
    valid64 = evaluate(model, base, args, device, seed, 64)
    valid32 = evaluate(model, base, args, device, seed, 32)
    metrics = {
        "seed": seed,
        "steps": step,
        "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "stored_bits": args.dim + 63 * args.detail_width,
        "raw_token_upper_bound_bits": 64 * math.ceil(math.log2(vocab)),
        "rate_fraction_of_token_upper_bound": (args.dim + 63 * args.detail_width) / (64 * math.ceil(math.log2(vocab))),
        "gradient_audit": audit,
        "valid64": valid64,
        "valid32_ood": valid32,
    }
    torch.save({"seed": seed, "config": vars(args), "model": model.state_dict()}, out / f"checkpoint_seed_{seed}.pt")
    (out / f"metrics_seed_{seed}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate(rows: List[dict], args) -> dict:
    modes = ("normal", "detail_shift", "detail_zero", "detail_wrong_sample", "root_zero")
    summary_metrics: Dict[str, dict] = {}
    for length_key in ("valid64", "valid32_ood"):
        summary_metrics[length_key] = {}
        for mode in modes:
            summary_metrics[length_key][mode] = {}
            for metric in ("nll", "token_top1", "sequence_exact"):
                values = [row[length_key]["metrics"][mode][metric] for row in rows]
                summary_metrics[length_key][mode][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    gradients = {}
    for name in ("write_grad_norm", "fold_grad_norm", "detail_grad_norm", "unfold_grad_norm", "read_grad_norm"):
        values = [row["gradient_audit"][name] for row in rows]
        gradients[name] = {"mean": float(np.mean(values)), "values": values}

    gates_per_seed = []
    for row in rows:
        normal = row["valid64"]["metrics"]["normal"]
        shift = row["valid64"]["metrics"]["detail_shift"]
        zero = row["valid64"]["metrics"]["detail_zero"]
        wrong = row["valid64"]["metrics"]["detail_wrong_sample"]
        ood = row["valid32_ood"]["metrics"]["normal"]
        audit = row["gradient_audit"]
        gates_per_seed.append({
            "P1_echo": normal["token_top1"] >= 0.90 and normal["sequence_exact"] >= 0.10,
            "P2_address_shift": normal["token_top1"] - shift["token_top1"] >= 0.50,
            "P3_detail_zero": normal["token_top1"] - zero["token_top1"] >= 0.50,
            "P4_wrong_sample": normal["token_top1"] - wrong["token_top1"] >= 0.50,
            "P5_length32": ood["token_top1"] >= 0.80,
            "P6_all_gradients": all(math.isfinite(audit[k]) and audit[k] > 0 for k in (
                "write_grad_norm", "fold_grad_norm", "detail_grad_norm", "unfold_grad_norm", "read_grad_norm"
            )),
            "P7_root_causal": normal["token_top1"] - row["valid64"]["metrics"]["root_zero"]["token_top1"] >= 0.10,
        })
    gates = {name: all(row[name] for row in gates_per_seed) for name in gates_per_seed[0]}
    return {
        "claim": "S3-TREEHEAP-CODEC-C01",
        "predict": "P-S3-TREEHEAP-CODEC-01",
        "host": socket.gethostname(),
        "config": vars(args),
        "aggregate": summary_metrics,
        "gradient_audit": gradients,
        "gates_per_seed": gates_per_seed,
        "gates": gates,
        "claim_supported": all(gates.values()),
        "boundary": "Native raw-token echo codec only; not semantics, world knowledge, reasoning, entropy-coded compression, or architecture superiority.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-source", default="ara/s3-generation/src/s3_residual_treeheap_forest_pretrain.py")
    ap.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_treeheap_native_codec")
    ap.add_argument("--seeds", default="71411,71412,71413")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--detail-width", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--max-train-blocks", type=int, default=1_000_000)
    ap.add_argument("--max-valid-blocks", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_module(Path(args.base_source))
    manifest = base.manifest(Path(args.block_dir), "train")
    vocab = int(manifest["tokenizer"]["vocab"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in (int(x) for x in args.seeds.split(",") if x.strip()):
        metrics_path = out / f"metrics_seed_{seed}.json"
        if metrics_path.exists():
            print(f"[resume] {metrics_path}", flush=True)
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            rows.append(run_seed(base, args, device, vocab, seed, out))
    summary = aggregate(rows, args)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Native TreeHeap Codec\n\n"
        f"Claim supported: `{summary['claim_supported']}`. See `summary.json`, per-seed metrics, traces, and checkpoints.\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": summary["gates"], "claim_supported": summary["claim_supported"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
