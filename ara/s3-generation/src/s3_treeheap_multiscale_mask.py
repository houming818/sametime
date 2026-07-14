#!/usr/bin/env python3
"""Compare ordinary echo with multiscale TreeHeap subheap-detail masking."""
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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_base(path):
    spec = importlib.util.spec_from_file_location("mask_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Codec(nn.Module):
    def __init__(self, vocab, dim, detail_width, hidden):
        super().__init__()
        self.write = nn.Embedding(vocab, dim)
        self.fold = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.LayerNorm(dim)
        )
        self.detail = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, detail_width)
        )
        self.unfold = nn.Sequential(
            nn.Linear(dim + detail_width, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim)
        )
        self.read = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(vocab))

    def encode(self, tokens):
        node = self.write(tokens)
        details = []
        while node.shape[1] > 1:
            left, right = node[:, 0::2], node[:, 1::2]
            details.append(self.detail((left - right) * math.sqrt(0.5)))
            node = self.fold((left + right) * math.sqrt(0.5))
        return node[:, 0], details

    def decode(self, root, details):
        current = root[:, None]
        for detail in reversed(details):
            left, right = self.unfold(torch.cat((current, detail), -1)).chunk(2, -1)
            current = torch.stack((left, right), 2).flatten(1, 2)
        return current

    def logits(self, leaves):
        return F.linear(self.read(leaves), self.write.weight, self.bias)

    def forward(self, tokens):
        root, details = self.encode(tokens)
        return self.logits(self.decode(root, details))


def batches(base, args, split, seed, batch, maximum):
    directory = Path(args.block_dir)
    yield from base.iter_batches(
        directory, base.manifest(directory, split), batch, seed, maximum
    )


def spans(batch, length, depth, generator, device):
    width = 1 << depth
    starts = (
        torch.randint(0, length // width, (batch,), generator=generator) * width
    ).to(device)
    pos = torch.arange(length, device=device)[None]
    selected = (pos >= starts[:, None]) & (pos < starts[:, None] + width)
    return starts, selected


def cut(details, starts, depth):
    output = list(details)
    for level in range(depth):
        value = details[level].clone()
        receptive = 1 << (level + 1)
        first = starts // receptive
        count = 1 << (depth - level - 1)
        index = first[:, None] + torch.arange(count, device=value.device)[None]
        value.scatter_(1, index[..., None].expand(-1, -1, value.shape[-1]), 0.0)
        output[level] = value
    return output


def parameter_grad_norm(module):
    values = [p.grad.float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def audit(model, tokens, depth, seed):
    model.train()
    model.zero_grad(set_to_none=True)
    root, details = model.encode(tokens)
    starts, selected = spans(
        len(tokens), tokens.shape[1], depth, torch.Generator().manual_seed(seed), tokens.device
    )
    logits = model.logits(model.decode(root, cut(details, starts, depth)))
    loss = F.cross_entropy(logits[selected], tokens[selected])
    loss.backward()
    result = {
        "loss": float(loss),
        "write_grad_norm": parameter_grad_norm(model.write),
        "fold_grad_norm": parameter_grad_norm(model.fold),
        "detail_grad_norm": parameter_grad_norm(model.detail),
        "unfold_grad_norm": parameter_grad_norm(model.unfold),
        "read_grad_norm": parameter_grad_norm(model.read),
    }
    model.zero_grad(set_to_none=True)
    return result


def train(name, base, args, device, vocab, seed, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    depth_rng = random.Random(seed + 1)
    mask_rng = torch.Generator().manual_seed(seed + 2)
    model = Codec(vocab, args.dim, args.detail_width, args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trace = out / ("trace_" + name + ".jsonl")
    trace.unlink(missing_ok=True)
    started = time.time()
    last = None
    step = 0
    for tokens, _ in batches(
        base, args, "train", seed, args.batch, args.max_train_blocks
    ):
        tokens = tokens.to(device)
        last = tokens
        optimizer.zero_grad(set_to_none=True)
        depth = 0
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            if name == "echo":
                logits = model(tokens)
                loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
            else:
                depth = depth_rng.randint(args.min_depth, args.max_depth)
                root, details = model.encode(tokens)
                starts, selected = spans(
                    len(tokens), tokens.shape[1], depth, mask_rng, device
                )
                logits = model.logits(model.decode(root, cut(details, starts, depth)))
                loss = F.cross_entropy(logits[selected], tokens[selected])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if not math.isfinite(float(loss)):
            raise FloatingPointError(name + " non-finite loss")
        if step == 1 or step % args.log_every == 0:
            row = {
                "model": name,
                "step": step,
                "blocks": min(step * args.batch, args.max_train_blocks),
                "depth": depth,
                "nll": float(loss),
                "grad_norm": float(norm),
                "elapsed_sec": time.time() - started,
                "gpu_memory_mb": (
                    torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
                ),
            }
            with trace.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    gradients = audit(
        model, last[: min(8, len(last))], args.max_depth, seed + 3
    )
    info = {
        "model": name,
        "steps": step,
        "elapsed_sec": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "gradient_audit": gradients,
    }
    torch.save(
        {"model": model.state_dict(), "config": vars(args), "name": name},
        out / ("checkpoint_" + name + ".pt"),
    )
    return model, info


@torch.no_grad()
def evaluate(model, name, base, args, device, seed):
    model.eval()
    modes = ("normal", "root_zero", "root_wrong")
    totals = {
        depth: {mode: {"loss": 0.0, "hit": 0, "count": 0} for mode in modes}
        for depth in range(args.min_depth, args.max_depth + 1)
    }
    generators = {
        depth: torch.Generator().manual_seed(seed + depth)
        for depth in range(args.min_depth, args.max_depth + 1)
    }
    echo = {"loss": 0.0, "hit": 0, "count": 0}
    examples = []
    for tokens, _ in batches(
        base, args, "valid", seed + 100, args.eval_batch, args.max_valid_blocks
    ):
        tokens = tokens.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            root, details = model.encode(tokens)
            full = model.logits(model.decode(root, details))
        echo["loss"] += float(
            F.cross_entropy(full.flatten(0, 1), tokens.flatten(), reduction="sum")
        )
        echo["hit"] += int(full.argmax(-1).eq(tokens).sum())
        echo["count"] += tokens.numel()
        for depth in totals:
            starts, selected = spans(
                len(tokens), tokens.shape[1], depth, generators[depth], device
            )
            damaged = cut(details, starts, depth)
            roots = {
                "normal": root,
                "root_zero": torch.zeros_like(root),
                "root_wrong": root.roll(1, 0),
            }
            predictions = None
            for mode, candidate in roots.items():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model.logits(model.decode(candidate, damaged))
                    loss = F.cross_entropy(
                        logits[selected], tokens[selected], reduction="sum"
                    )
                row = totals[depth][mode]
                row["loss"] += float(loss)
                row["hit"] += int((logits.argmax(-1).eq(tokens) & selected).sum())
                row["count"] += int(selected.sum())
                if mode == "normal":
                    predictions = logits.argmax(-1)
            if depth == args.max_depth and len(examples) < 6:
                start = int(starts[0])
                end = start + (1 << depth)
                examples.append(
                    {
                        "span": [start, end],
                        "gold_ids": tokens[0, start:end].cpu().tolist(),
                        "pred_ids": predictions[0, start:end].cpu().tolist(),
                    }
                )
    depth_rows = {}
    for depth, mode_rows in totals.items():
        depth_rows[str(depth)] = {}
        for mode, row in mode_rows.items():
            depth_rows[str(depth)][mode] = {
                "nll": row["loss"] / row["count"],
                "token_top1": row["hit"] / row["count"],
            }
        depth_rows[str(depth)]["root_nll_delta"] = (
            depth_rows[str(depth)]["root_zero"]["nll"]
            - depth_rows[str(depth)]["normal"]["nll"]
        )
        depth_rows[str(depth)]["wrong_root_nll_delta"] = (
            depth_rows[str(depth)]["root_wrong"]["nll"]
            - depth_rows[str(depth)]["normal"]["nll"]
        )
    return {
        "model": name,
        "echo": {
            "nll": echo["loss"] / echo["count"],
            "token_top1": echo["hit"] / echo["count"],
        },
        "depths": depth_rows,
        "examples_depth_max": examples,
    }


def spearman(values):
    ranks = np.argsort(np.argsort(np.asarray(values)))
    return float(np.corrcoef(np.arange(len(values)), ranks)[0, 1])


def summary(results, args):
    echo = results["echo"]["evaluation"]
    masked = results["multiscale_mask"]["evaluation"]
    keys = [str(depth) for depth in range(args.min_depth, args.max_depth + 1)]
    gains = {
        key: echo["depths"][key]["normal"]["nll"]
        - masked["depths"][key]["normal"]["nll"]
        for key in keys
    }
    deltas = [masked["depths"][key]["root_nll_delta"] for key in keys]
    deepest = str(args.max_depth)
    gradient = results["multiscale_mask"]["train"]["gradient_audit"]
    gates = {
        "P1_deep_mask_gain": all(
            gains[str(depth)] >= 0.30
            for depth in (args.max_depth - 1, args.max_depth)
        ),
        "P2_root_causal_depth_max": masked["depths"][deepest]["root_nll_delta"] >= 0.10,
        "P3_positive_depth_trend": spearman(deltas) >= 0.60,
        "P4_more_root_than_echo": (
            masked["depths"][deepest]["root_nll_delta"]
            - echo["depths"][deepest]["root_nll_delta"]
            >= 0.05
        ),
        "P5_wrong_root_causal": masked["depths"][deepest]["wrong_root_nll_delta"] >= 0.10,
        "P6_all_gradients": all(
            math.isfinite(gradient[key]) and gradient[key] > 0
            for key in (
                "write_grad_norm",
                "fold_grad_norm",
                "detail_grad_norm",
                "unfold_grad_norm",
                "read_grad_norm",
            )
        ),
    }
    return {
        "claim": "S3-TREEHEAP-MASK-C01",
        "predict": "P-S3-TREEHEAP-MASK-01",
        "host": socket.gethostname(),
        "config": vars(args),
        "models": results,
        "derived": {
            "masked_nll_improvement_over_echo": gains,
            "multiscale_root_nll_deltas": deltas,
            "root_delta_depth_spearman": spearman(deltas),
        },
        "gates": gates,
        "claim_supported_smoke": all(gates.values()),
        "boundary": (
            "Compression-style upward extraction only. The encoder observed the "
            "complete block; this is not inference of unseen text or world knowledge."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-source",
        default="ara/s3-generation/src/s3_residual_treeheap_forest_pretrain.py",
    )
    parser.add_argument(
        "--block-dir",
        default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64",
    )
    parser.add_argument(
        "--evidence-dir",
        default="ara/s3-generation/evidence/s3_treeheap_multiscale_mask/smoke",
    )
    parser.add_argument("--seed", type=int, default=71421)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--detail-width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--max-train-blocks", type=int, default=100000)
    parser.add_argument("--max-valid-blocks", type=int, default=512)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_base(Path(args.base_source))
    vocab = int(base.manifest(Path(args.block_dir), "train")["tokenizer"]["vocab"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}
    for name in ("echo", "multiscale_mask"):
        model, train_info = train(
            name, base, args, device, vocab, args.seed, out
        )
        evaluation = evaluate(
            model, name, base, args, device, args.seed
        )
        results[name] = {"train": train_info, "evaluation": evaluation}
        (out / ("metrics_" + name + ".json")).write_text(
            json.dumps(results[name], indent=2), encoding="utf-8"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final = summary(results, args)
    (out / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Multiscale Subheap Mask Smoke\n\n"
        + "Smoke supported: "
        + str(final["claim_supported_smoke"])
        + ". See summary.json.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gates": final["gates"],
                "derived": final["derived"],
                "claim_supported_smoke": final["claim_supported_smoke"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
