#!/usr/bin/env python3
"""Repair addressed TreeHeap residuals after complete-input FOLD."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_annealed_frontier_pretrain as anneal
import s3_wmt_treeheap_seq2seq as base


class RepairKernel(nn.Module):
    """A shared residual predictor; cross-scale inputs are an explicit ablation."""

    def __init__(self, dim: int, depths: int, hidden: int, cross_scale: bool):
        super().__init__()
        self.cross_scale = cross_scale
        self.depth = nn.Embedding(depths, 16)
        feature_dim = dim + 16
        if cross_scale:
            feature_dim += 3 * dim + 4  # root, left/right residual neighbor, path
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    @staticmethod
    def path_features(batch: int, width: int, device: torch.device, reverse: bool) -> torch.Tensor:
        if width == 1:
            x = torch.zeros(1, device=device)
        else:
            x = torch.linspace(0.0, 1.0, width, device=device)
        if reverse:
            x = 1.0 - x
        feature = torch.stack((x, x.square(), torch.sin(math.pi * x), torch.cos(math.pi * x)), -1)
        return feature[None].expand(batch, -1, -1)

    def forward(
        self,
        parent: torch.Tensor,
        root: torch.Tensor,
        damaged_detail: torch.Tensor,
        depth: int,
        wrong_address: bool = False,
    ) -> torch.Tensor:
        batch, width, _ = parent.shape
        depth_id = torch.full((batch, width), depth, dtype=torch.long, device=parent.device)
        features = [parent, self.depth(depth_id)]
        if self.cross_scale:
            left = F.pad(damaged_detail[:, :-1], (0, 0, 1, 0))
            right = F.pad(damaged_detail[:, 1:], (0, 0, 0, 1))
            if wrong_address:
                left, right = right.roll(1, dims=1), left.roll(-1, dims=1)
            features.extend(
                (
                    root[:, None].expand(-1, width, -1),
                    left,
                    right,
                    self.path_features(batch, width, parent.device, wrong_address),
                )
            )
        return self.net(torch.cat(features, -1))


def make_model(checkpoint: dict, device: str) -> anneal.FrontierModel:
    config = checkpoint["config"]
    model = anneal.FrontierModel(
        config["vocab"], config["dim"], config["hidden"],
        config["heap_width"], config["pad"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def loader(args, split: str, seed: int) -> DataLoader:
    return anneal.make_loader(args, split, seed)


def damage_mask(batch: int, width: int, severity: str, rng: random.Random, device: torch.device) -> torch.Tensor:
    if severity == "single":
        count = 1
    elif severity == "quarter":
        count = max(1, math.ceil(width * 0.25))
    elif severity == "half":
        count = max(1, math.ceil(width * 0.50))
    else:
        raise ValueError(severity)
    mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
    for row in range(batch):
        start = rng.randrange(width - count + 1)
        mask[row, start : start + count] = True
    return mask


def state_for_depth(model: anneal.FrontierModel, source: torch.Tensor, depth: int):
    length = torch.full((source.shape[0],), source.shape[1], dtype=torch.long, device=source.device)
    leaf, root, details, fold_masks = model.encoder.fold(source, length)
    levels, _ = model.encoder.unfold(root, details, fold_masks)
    parent = levels[model.depths - depth - 1]
    return leaf, root, details, levels, fold_masks, parent


def normalized_target_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = prediction[mask]
    gold = target[mask]
    scale = gold.square().mean().detach().clamp_min(1e-6)
    return (pred - gold).square().mean() / scale


def train(args, model: anneal.FrontierModel, output: Path):
    parent_only = RepairKernel(args.dim, model.depths, args.repair_hidden, False).to(args.device)
    cross_scale = RepairKernel(args.dim, model.depths, args.repair_hidden, True).to(args.device)
    parameters = list(parent_only.parameters()) + list(cross_scale.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    batches = iter(loader(args, "train", args.seed + 100))
    rng = random.Random(args.seed + 901)
    trace = []
    started = time.time()
    finite = True
    for step in range(1, args.steps + 1):
        source, _ = next(batches)
        source = source.to(args.device)
        depth = rng.randrange(model.depths)
        severity = rng.choice(("single", "quarter"))
        with torch.no_grad():
            _, root, details, _, _, parent = state_for_depth(model, source, depth)
            target = details[depth]
        mask = damage_mask(source.shape[0], target.shape[1], severity, rng, source.device)
        damaged = target.masked_fill(mask[:, :, None], 0.0)
        prediction_parent = parent_only(parent, root, damaged, depth)
        prediction_cross = cross_scale(parent, root, damaged, depth)
        loss_parent = normalized_target_loss(prediction_parent, target, mask)
        loss_cross = normalized_target_loss(prediction_cross, target, mask)
        loss = loss_parent + loss_cross
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "depth": depth,
                "severity": severity,
                "parent_normalized_mse": float(loss_parent.detach()),
                "cross_normalized_mse": float(loss_cross.detach()),
                "elapsed_sec": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)
    torch.save(
        {
            "parent_only": parent_only.state_dict(),
            "cross_scale": cross_scale.state_dict(),
            "config": vars(args),
            "trace": trace,
        },
        output / "repair_kernels.pt",
    )
    return parent_only, cross_scale, trace, finite, time.time() - started


def replace_detail(details: Sequence[torch.Tensor], depth: int, mask: torch.Tensor, value: torch.Tensor) -> List[torch.Tensor]:
    result = list(details)
    result[depth] = torch.where(mask[:, :, None], value, details[depth])
    return result


def unfold_leaf(model: anneal.FrontierModel, root, details, masks):
    levels, level_masks = model.encoder.unfold(root, details, masks)
    return levels[-1], level_masks[-1]


def affected_leaf_mask(mask: torch.Tensor, depth: int) -> torch.Tensor:
    return mask.repeat_interleave(2 ** (depth + 1), dim=1)


@torch.no_grad()
def token_accuracy(states: torch.Tensor, source: torch.Tensor, affected: torch.Tensor, embedding: torch.Tensor, chunk: int = 512) -> float:
    rows = states[affected]
    target = source[affected]
    if rows.numel() == 0:
        return 0.0
    rows = F.normalize(rows, dim=-1)
    weight = F.normalize(embedding, dim=-1)
    hits = 0
    for start in range(0, rows.shape[0], chunk):
        logits = rows[start : start + chunk] @ weight.T
        hits += int(logits.argmax(-1).eq(target[start : start + chunk]).sum())
    return hits / rows.shape[0]


@torch.no_grad()
def future_nll(model: anneal.FrontierModel, leaf, valid, target, bos: int) -> float:
    nodes = leaf + model.resolution.weight[model.depths]
    logits = model.decoder.teacher(nodes, valid, target, bos)
    return float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="mean"))


def average(rows: Iterable[dict], key: str) -> float:
    values = [row[key] for row in rows]
    return sum(values) / max(1, len(values))


@torch.no_grad()
def evaluate_combo(args, model, parent_only, cross_scale, sp, depth: int, severity: str) -> dict:
    parent_only.eval(); cross_scale.eval()
    rows = []
    rng = random.Random(args.seed + 5000 + 100 * depth + sum(map(ord, severity)))
    eval_loader = loader(args, "test", args.seed + 2000 + depth)
    embedding = model.encoder.embedding.weight
    for batch_no, (source, target) in enumerate(eval_loader, 1):
        source, target = source.to(args.device), target.to(args.device)
        clean_leaf, root, details, _, masks, parent = state_for_depth(model, source, depth)
        clean_detail = details[depth]
        mask = damage_mask(source.shape[0], clean_detail.shape[1], severity, rng, source.device)
        zero = torch.zeros_like(clean_detail)
        damaged_detail = torch.where(mask[:, :, None], zero, clean_detail)
        pred_parent = parent_only(parent, root, damaged_detail, depth)
        pred_cross = cross_scale(parent, root, damaged_detail, depth)
        pred_wrong = cross_scale(parent, root, damaged_detail, depth, wrong_address=True)
        leaf_damage, valid = unfold_leaf(model, root, replace_detail(details, depth, mask, zero), masks)
        leaf_parent, _ = unfold_leaf(model, root, replace_detail(details, depth, mask, pred_parent), masks)
        leaf_cross, _ = unfold_leaf(model, root, replace_detail(details, depth, mask, pred_cross), masks)
        leaf_wrong, _ = unfold_leaf(model, root, replace_detail(details, depth, mask, pred_wrong), masks)
        affected = affected_leaf_mask(mask, depth) & valid
        target_energy = clean_detail[mask].square().mean().clamp_min(1e-9)
        damage_leaf_mse = (leaf_damage[affected] - clean_leaf[affected]).square().mean()
        parent_leaf_mse = (leaf_parent[affected] - clean_leaf[affected]).square().mean()
        cross_leaf_mse = (leaf_cross[affected] - clean_leaf[affected]).square().mean()
        wrong_leaf_mse = (leaf_wrong[affected] - clean_leaf[affected]).square().mean()
        row = {
            "detail_zero_nmse": float(clean_detail[mask].square().mean() / target_energy),
            "detail_parent_nmse": float((pred_parent[mask] - clean_detail[mask]).square().mean() / target_energy),
            "detail_cross_nmse": float((pred_cross[mask] - clean_detail[mask]).square().mean() / target_energy),
            "detail_wrong_nmse": float((pred_wrong[mask] - clean_detail[mask]).square().mean() / target_energy),
            "leaf_damage_mse": float(damage_leaf_mse),
            "leaf_parent_mse": float(parent_leaf_mse),
            "leaf_cross_mse": float(cross_leaf_mse),
            "leaf_wrong_mse": float(wrong_leaf_mse),
            "token_clean": token_accuracy(clean_leaf, source, affected, embedding),
            "token_damage": token_accuracy(leaf_damage, source, affected, embedding),
            "token_parent": token_accuracy(leaf_parent, source, affected, embedding),
            "token_cross": token_accuracy(leaf_cross, source, affected, embedding),
            "nll_clean": future_nll(model, clean_leaf, valid, target, args.bos),
            "nll_damage": future_nll(model, leaf_damage, valid, target, args.bos),
            "nll_parent": future_nll(model, leaf_parent, valid, target, args.bos),
            "nll_cross": future_nll(model, leaf_cross, valid, target, args.bos),
            "nll_wrong": future_nll(model, leaf_wrong, valid, target, args.bos),
        }
        rows.append(row)
        if batch_no >= args.eval_batches:
            break
    metrics = {key: average(rows, key) for key in rows[0]}
    damage = metrics["leaf_damage_mse"]
    metrics["latent_repair_parent"] = 1.0 - metrics["leaf_parent_mse"] / max(1e-12, damage)
    metrics["latent_repair_cross"] = 1.0 - metrics["leaf_cross_mse"] / max(1e-12, damage)
    nll_damage = metrics["nll_damage"] - metrics["nll_clean"]
    metrics["nll_damage_delta"] = nll_damage
    metrics["nll_repair_parent"] = (metrics["nll_damage"] - metrics["nll_parent"]) / max(1e-12, nll_damage) if nll_damage > 0 else 0.0
    metrics["nll_repair_cross"] = (metrics["nll_damage"] - metrics["nll_cross"]) / max(1e-12, nll_damage) if nll_damage > 0 else 0.0
    metrics["wrong_address_mse_ratio"] = metrics["leaf_wrong_mse"] / max(1e-12, metrics["leaf_cross_mse"])
    return {"depth": depth, "severity": severity, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/nas/ara/s3-generation/evidence/s3_annealed_frontier_pretrain/checkpoint_annealed.pt")
    parser.add_argument("--root", default="/home/nio/datasets/pretrain")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_multiresolution_damage_repair")
    parser.add_argument("--seed", type=int, default=73101)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--repair-hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    for key in ("context", "future", "heap_width", "dim", "hidden", "pad", "vocab", "bos", "eos"):
        setattr(args, key, config[key])
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    output = Path(args.evidence_dir); output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    model = make_model(checkpoint, args.device)
    before = model_digest(model)
    parent_only, cross_scale, trace, finite, train_seconds = train(args, model, output)
    results = []
    for depth in range(model.depths):
        for severity in ("single", "quarter", "half"):
            row = evaluate_combo(args, model, parent_only, cross_scale, sp, depth, severity)
            results.append(row); print(json.dumps(row), flush=True)
    after = model_digest(model)
    mean_latent_cross = average(results, "latent_repair_cross")
    mean_latent_parent = average(results, "latent_repair_parent")
    address_ratio = average(results, "wrong_address_mse_ratio")
    causal_nll = [row for row in results if row["nll_damage_delta"] >= 0.01]
    nll_repair = average(causal_nll, "nll_repair_cross") if causal_nll else 0.0
    token_drop = average(results, "token_clean") - average(results, "token_damage")
    gates = {
        "P1_damage_real": average(results, "leaf_damage_mse") > 1e-4 and token_drop >= 0.05,
        "P2_cross_repairs_half": mean_latent_cross >= 0.50,
        "P3_cross_beats_parent": mean_latent_cross - mean_latent_parent >= 0.05,
        "P4_address_causal": address_ratio >= 1.10,
        "P5_language_recovery": bool(causal_nll) and nll_repair >= 0.25,
        "P6_graceful": finite and all(row["leaf_cross_mse"] < row["leaf_damage_mse"] for row in results),
        "P7_frozen": before == after,
    }
    derived = {
        "mean_leaf_damage_mse": average(results, "leaf_damage_mse"),
        "mean_token_drop": token_drop,
        "mean_latent_repair_parent": mean_latent_parent,
        "mean_latent_repair_cross": mean_latent_cross,
        "cross_gain_over_parent": mean_latent_cross - mean_latent_parent,
        "mean_wrong_address_mse_ratio": address_ratio,
        "causal_nll_cells": len(causal_nll),
        "mean_nll_repair_cross_on_causal_cells": nll_repair,
        "base_digest_before": before,
        "base_digest_after": after,
    }
    summary = {
        "claim": "S3-MULTIRES-REPAIR-C01",
        "host": socket.gethostname(),
        "config": vars(args),
        "train_seconds": train_seconds,
        "trace": trace,
        "results": results,
        "derived": derived,
        "gates": gates,
        "decision": "supported" if all(gates.values()) else "partial or rejected; inspect gates",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text(
        "# Multiresolution Damage Repair\n\nComplete input is FOLDed before addressed residual damage. The annealed base is frozen.\n",
        encoding="utf-8",
    )
    print(json.dumps({"derived": derived, "gates": gates, "decision": summary["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
