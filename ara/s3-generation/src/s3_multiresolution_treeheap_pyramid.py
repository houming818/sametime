#!/usr/bin/env python3
"""Frozen-root multiresolution TreeHeap pyramid experiment.

The pretrained TreeHeap encoder is immutable. Learned shared detail codecs
store one k-dimensional code at every internal address and recursively decode
the original 64 leaf states. Matched-rate flat and fixed Haar controls prevent
the rate-distortion result from being attributed to capacity alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


WIDTHS = (0, 8, 16, 32, 64)


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("residual_forest_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base model from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class DetailCodec(nn.Module):
    """One shared analysis/synthesis kernel reused at every tree address."""

    def __init__(self, dim: int, width: int, hidden: int):
        super().__init__()
        self.width = width
        self.analysis = None if width == 0 else nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.synthesis = nn.Sequential(
            nn.Linear(dim + width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim),
        )

    def detail(self, left: torch.Tensor, right: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        if self.analysis is None:
            return parent.new_zeros(*parent.shape[:-1], 0)
        return self.analysis(torch.cat((left, right, parent), dim=-1))

    def split(self, parent: torch.Tensor, detail: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        children = self.synthesis(torch.cat((parent, detail), dim=-1))
        return children.chunk(2, dim=-1)


class FlatCodec(nn.Module):
    """Structure-blind latent attention codec at a matched-or-higher rate."""

    def __init__(self, dim: int, code_size: int, heads: int = 4):
        super().__init__()
        self.latent_count = math.ceil(code_size / dim)
        self.stored_floats = self.latent_count * dim
        self.position = nn.Parameter(torch.empty(64, dim))
        self.latent = nn.Parameter(torch.empty(self.latent_count, dim))
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.write = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.read_norm = nn.LayerNorm(dim)
        self.write_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.latent, std=0.02)

    def forward(self, leaves: torch.Tensor) -> torch.Tensor:
        batch = leaves.shape[0]
        source = leaves + self.position[None]
        query = self.latent[None].expand(batch, -1, -1)
        code, _ = self.read(query, source, source, need_weights=False)
        code = self.read_norm(code + query)
        output_query = self.position[None].expand(batch, -1, -1)
        output, _ = self.write(output_query, code, code, need_weights=False)
        return self.write_norm(output + output_query)


@torch.no_grad()
def frozen_tree(model: nn.Module, tokens: torch.Tensor):
    """Return leaves, root, and true (left,right,parent) states bottom-up."""
    node = model.embedding(tokens)
    leaves = node
    levels = []
    while node.shape[1] > 1:
        if node.shape[1] % 2:
            node = torch.cat((node, torch.zeros_like(node[:, :1])), dim=1)
        left, right = node[:, 0::2], node[:, 1::2]
        state = (left + right) * math.sqrt(0.5)
        for _ in range(model.refinements):
            delta, _, _ = model.kernel(state, left, right)
            state = model.norm(delta)
        levels.append((left, right, state))
        node = state
    return leaves, node[:, 0], levels


def pyramid_decode(codec: DetailCodec, root: torch.Tensor, levels, destroy: bool = False):
    details = [codec.detail(left, right, parent) for left, right, parent in levels]
    current = root[:, None]
    for level_index in range(len(levels) - 1, -1, -1):
        detail = details[level_index]
        if destroy and detail.shape[1] > 1:
            detail = detail.roll(shifts=1, dims=1)
        left, right = codec.split(current, detail)
        current = torch.stack((left, right), dim=2).flatten(1, 2)
    return current


def fixed_haar(leaves: torch.Tensor, width: int) -> torch.Tensor:
    """Orthonormal mean/detail pyramid, truncating each addressed detail."""
    node = leaves
    details = []
    while node.shape[1] > 1:
        left, right = node[:, 0::2], node[:, 1::2]
        parent = (left + right) * math.sqrt(0.5)
        detail = (left - right) * math.sqrt(0.5)
        if width < detail.shape[-1]:
            detail = torch.cat((detail[..., :width], torch.zeros_like(detail[..., width:])), dim=-1)
        details.append(detail)
        node = parent
    current = node
    for detail in reversed(details):
        left = (current + detail) * math.sqrt(0.5)
        right = (current - detail) * math.sqrt(0.5)
        current = torch.stack((left, right), dim=2).flatten(1, 2)
    return current


def mean_only(leaves: torch.Tensor) -> torch.Tensor:
    return leaves.mean(dim=1, keepdim=True).expand_as(leaves)


def batch_stream(base, block_dir: Path, split: str, batch: int, seed: int, max_blocks: int):
    man = base.manifest(block_dir, split)
    yield from base.iter_batches(block_dir, man, batch, seed, max_blocks)


@torch.no_grad()
def root_nll(model: nn.Module, batches, device: torch.device) -> float:
    total = 0.0
    count = 0
    model.eval()
    for tokens, target in batches:
        tokens, target = tokens.to(device), target.to(device)
        logits, _, _ = model(tokens)
        total += float(F.cross_entropy(logits, target, reduction="sum"))
        count += target.numel()
    return total / max(1, count)


@torch.no_grad()
def token_metrics(recon: torch.Tensor, tokens: torch.Tensor, embedding: torch.Tensor) -> dict:
    # The embedding table is the frozen encoder's own coordinate system.
    q = F.normalize(recon, dim=-1).flatten(0, 1)
    table = F.normalize(embedding, dim=-1)
    top1_parts = []
    top5_parts = []
    chunk = 2048
    for start in range(0, q.shape[0], chunk):
        top = (q[start:start + chunk] @ table.T).topk(5, dim=-1).indices
        top1_parts.append(top[:, 0])
        top5_parts.append(top)
    top1 = torch.cat(top1_parts).view_as(tokens)
    top5 = torch.cat(top5_parts).view(*tokens.shape, 5)
    exact = top1.eq(tokens)
    bag_scores = []
    for pred, gold in zip(top1.tolist(), tokens.tolist()):
        p, g = {}, {}
        for x in pred:
            p[x] = p.get(x, 0) + 1
        for x in gold:
            g[x] = g.get(x, 0) + 1
        bag_scores.append(sum(min(v, g.get(k, 0)) for k, v in p.items()) / len(gold))
    return {
        "token_top1": float(exact.float().mean()),
        "token_top5": float(top5.eq(tokens[..., None]).any(-1).float().mean()),
        "sequence_exact": float(exact.all(-1).float().mean()),
        "bag_overlap": float(np.mean(bag_scores)),
    }


def evaluate(models, flat_models, frozen, base, args, device, seed: int) -> dict:
    for model in list(models.values()) + list(flat_models.values()):
        model.eval()
    sums: Dict[str, Dict[int, float]] = {
        name: {width: 0.0 for width in WIDTHS}
        for name in ("tree_mse", "destroy_mse", "flat_mse", "haar_mse")
    }
    mean_sum = 0.0
    count = 0
    token_cache = {width: [] for width in WIDTHS}
    token_ids = []
    with torch.no_grad():
        for tokens, _ in batch_stream(base, Path(args.block_dir), "valid", args.eval_batch, seed + 9000, args.max_valid_blocks):
            tokens = tokens.to(device)
            leaves, root, levels = frozen_tree(frozen, tokens)
            n = tokens.numel()
            mean_sum += float(F.mse_loss(mean_only(leaves), leaves, reduction="sum"))
            count += leaves.numel()
            for width in WIDTHS:
                recon = pyramid_decode(models[width], root, levels)
                destroyed = pyramid_decode(models[width], root, levels, destroy=True)
                flat = flat_models[width](leaves)
                haar = fixed_haar(leaves, width)
                sums["tree_mse"][width] += float(F.mse_loss(recon, leaves, reduction="sum"))
                sums["destroy_mse"][width] += float(F.mse_loss(destroyed, leaves, reduction="sum"))
                sums["flat_mse"][width] += float(F.mse_loss(flat, leaves, reduction="sum"))
                sums["haar_mse"][width] += float(F.mse_loss(haar, leaves, reduction="sum"))
                if sum(x.shape[0] for x in token_cache[width]) < args.token_metric_blocks:
                    token_cache[width].append(recon.cpu())
            if sum(x.shape[0] for x in token_ids) < args.token_metric_blocks:
                token_ids.append(tokens.cpu())
    rows = {}
    gold = torch.cat(token_ids)[:args.token_metric_blocks].to(device)
    emb = frozen.embedding.weight
    for width in WIDTHS:
        recon = torch.cat(token_cache[width])[:args.token_metric_blocks].to(device)
        rate = args.dim + 63 * width
        rows[str(width)] = {
            "stored_floats": rate,
            "rate_fraction": rate / (64 * args.dim),
            "tree_mse": sums["tree_mse"][width] / count,
            "destroy_mse": sums["destroy_mse"][width] / count,
            "flat_mse": sums["flat_mse"][width] / count,
            "flat_stored_floats": flat_models[width].stored_floats,
            "haar_mse": sums["haar_mse"][width] / count,
            **token_metrics(recon, gold, emb),
        }
    return {"widths": rows, "mean_only_mse": mean_sum / count}


def run_seed(base, frozen, args, device, seed: int, out: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    models = {w: DetailCodec(args.dim, w, args.codec_hidden).to(device) for w in WIDTHS}
    flat_models = {
        w: FlatCodec(args.dim, args.dim + 63 * w).to(device) for w in WIDTHS
    }
    params = list(p for m in models.values() for p in m.parameters())
    flat_params = list(p for m in flat_models.values() for p in m.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    flat_optimizer = torch.optim.AdamW(flat_params, lr=args.lr, weight_decay=args.weight_decay)
    trace_path = out / f"trace_seed_{seed}.jsonl"
    trace_path.unlink(missing_ok=True)
    started = time.time()
    step = 0
    frozen.eval()
    for tokens, _ in batch_stream(base, Path(args.block_dir), "train", args.batch, seed, args.max_train_blocks):
        tokens = tokens.to(device, non_blocking=True)
        with torch.no_grad():
            leaves, root, levels = frozen_tree(frozen, tokens)
        optimizer.zero_grad(set_to_none=True)
        flat_optimizer.zero_grad(set_to_none=True)
        tree_losses, flat_losses = [], []
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for width in WIDTHS:
                tree_losses.append(F.mse_loss(pyramid_decode(models[width], root, levels), leaves))
                flat_losses.append(F.mse_loss(flat_models[width](leaves), leaves))
            tree_loss = torch.stack(tree_losses).sum()
            flat_loss = torch.stack(flat_losses).sum()
        tree_loss.backward()
        flat_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        torch.nn.utils.clip_grad_norm_(flat_params, 1.0)
        optimizer.step()
        flat_optimizer.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "blocks": step * args.batch,
                "elapsed_sec": time.time() - started,
                "tree_mse": {str(w): float(v.detach()) for w, v in zip(WIDTHS, tree_losses)},
                "flat_mse": {str(w): float(v.detach()) for w, v in zip(WIDTHS, flat_losses)},
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
            }
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"seed": seed, **row}), flush=True)
    torch.save({
        "seed": seed,
        "detail_codecs": {str(k): v.state_dict() for k, v in models.items()},
        "flat_codecs": {str(k): v.state_dict() for k, v in flat_models.items()},
    }, out / f"checkpoint_seed_{seed}.pt")
    metrics = evaluate(models, flat_models, frozen, base, args, device, seed)
    metrics.update({
        "seed": seed,
        "steps": step,
        "elapsed_sec": time.time() - started,
        "detail_parameters": {str(k): sum(p.numel() for p in v.parameters()) for k, v in models.items()},
        "flat_parameters": {str(k): sum(p.numel() for p in v.parameters()) for k, v in flat_models.items()},
    })
    (out / f"metrics_seed_{seed}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def spearman_monotonic(values: Sequence[float]) -> float:
    # Widths are already ordered; reconstruction quality is -MSE.
    x = np.arange(len(values), dtype=np.float64)
    y = np.argsort(np.argsort(-np.asarray(values, dtype=np.float64))).astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def aggregate(seed_rows: List[dict], before_nll: float, after_nll: float, args, checkpoint: Path, fingerprint_before: str, fingerprint_after: str) -> dict:
    aggregate_widths = {}
    for width in WIDTHS:
        key = str(width)
        aggregate_widths[key] = {}
        for metric in ("tree_mse", "destroy_mse", "flat_mse", "haar_mse", "token_top1", "token_top5", "sequence_exact", "bag_overlap"):
            vals = [row["widths"][key][metric] for row in seed_rows]
            aggregate_widths[key][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        aggregate_widths[key]["stored_floats"] = seed_rows[0]["widths"][key]["stored_floats"]
        aggregate_widths[key]["flat_stored_floats"] = seed_rows[0]["widths"][key]["flat_stored_floats"]
        aggregate_widths[key]["rate_fraction"] = seed_rows[0]["widths"][key]["rate_fraction"]
    per_seed_monotonic = []
    per_seed_spearman = []
    for row in seed_rows:
        mses = [row["widths"][str(w)]["tree_mse"] for w in WIDTHS]
        per_seed_monotonic.append(all(b <= a for a, b in zip(mses, mses[1:])))
        per_seed_spearman.append(spearman_monotonic(mses))
    root_mse = aggregate_widths["0"]["tree_mse"]["mean"]
    k64_mse = aggregate_widths["64"]["tree_mse"]["mean"]
    destroy64 = aggregate_widths["64"]["destroy_mse"]["mean"]
    token64 = aggregate_widths["64"]["token_top1"]["mean"]
    destroy_token_values = []
    # The preregistered P5 can pass on MSE; destroyed token decoding is not needed.
    gates = {
        "P1_root_nll_unchanged": abs(after_nll - before_nll) < 1e-5 and fingerprint_before == fingerprint_after,
        "P2_mse_monotonic_all_seeds": all(per_seed_monotonic),
        "P3_spearman_ge_0_9_all_seeds": all(x >= 0.9 for x in per_seed_spearman),
        "P4_k64_halves_root_mse": k64_mse <= 0.5 * root_mse,
        "P5_address_destruction_ge_10pct": destroy64 >= 1.10 * k64_mse,
        "P6_beats_flat_or_haar_equal_rate": any(
            aggregate_widths[str(w)]["tree_mse"]["mean"] < min(
                aggregate_widths[str(w)]["flat_mse"]["mean"],
                aggregate_widths[str(w)]["haar_mse"]["mean"],
            ) for w in WIDTHS
        ),
    }
    return {
        "claim": "S3-TREEHEAP-PYRAMID-C01",
        "predict": "P-S3-TREEHEAP-PYRAMID-01",
        "host": socket.gethostname(),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "config": vars(args),
        "frozen_root": {
            "nll_before": before_nll,
            "nll_after": after_nll,
            "abs_delta": abs(after_nll - before_nll),
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
        },
        "aggregate": aggregate_widths,
        "per_seed_monotonic": per_seed_monotonic,
        "per_seed_spearman": per_seed_spearman,
        "gates": gates,
        "mechanism_supported": all(gates[k] for k in (
            "P1_root_nll_unchanged", "P2_mse_monotonic_all_seeds",
            "P3_spearman_ge_0_9_all_seeds", "P4_k64_halves_root_mse",
            "P5_address_destruction_ge_10pct",
        )),
        "treeheap_advantage_supported": gates["P6_beats_flat_or_haar_equal_rate"],
        "boundary": "Activation rate-distortion proof only; not entropy-coded compression, semantic scale specialization, world knowledge, or architecture superiority.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-source", default="ara/s3-generation/src/s3_residual_treeheap_forest_pretrain.py")
    ap.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_residual_treeheap_forest/full/checkpoint_latest.pt")
    ap.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    ap.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_multiresolution_treeheap_pyramid")
    ap.add_argument("--seeds", default="71401,71402,71403")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--refinements", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--max-train-blocks", type=int, default=1_000_000)
    ap.add_argument("--max-valid-blocks", type=int, default=8192)
    ap.add_argument("--token-metric-blocks", type=int, default=256)
    ap.add_argument("--codec-hidden", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_module(Path(args.base_source))
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = payload["models"]["noresidual_forest"]["embedding.weight"].shape[0]
    frozen = base.TreeHeapForestLM(vocab, args.dim, args.rank, args.heads, args.refinements, False).to(device)
    frozen.load_state_dict(payload["models"]["noresidual_forest"])
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    audit_batches = list(batch_stream(base, Path(args.block_dir), "valid", args.eval_batch, 99001, min(args.max_valid_blocks, 2048)))
    fingerprint_before = tensor_fingerprint(frozen)
    nll_before = root_nll(frozen, audit_batches, device)
    seed_rows = []
    for seed in (int(x) for x in args.seeds.split(",") if x.strip()):
        metrics_path = out / f"metrics_seed_{seed}.json"
        if metrics_path.exists():
            print(f"[resume] loading {metrics_path}", flush=True)
            seed_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            seed_rows.append(run_seed(base, frozen, args, device, seed, out))
    nll_after = root_nll(frozen, audit_batches, device)
    fingerprint_after = tensor_fingerprint(frozen)
    summary = aggregate(seed_rows, nll_before, nll_after, args, checkpoint, fingerprint_before, fingerprint_after)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Multiresolution TreeHeap Pyramid\n\n"
        "See `summary.json`, per-seed metrics, traces, and checkpoints.\n\n"
        f"Mechanism supported: `{summary['mechanism_supported']}`.  \n"
        f"TreeHeap equal-rate advantage supported: `{summary['treeheap_advantage_supported']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": summary["gates"], "mechanism_supported": summary["mechanism_supported"], "treeheap_advantage_supported": summary["treeheap_advantage_supported"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
