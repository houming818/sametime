#!/usr/bin/env python3
"""Audit whether learned WMT fold histories depend on source content."""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
from s3_wmt_learned_fold_seq2seq import LearnedFoldTreeHeapSeq2Seq


def normalized_entropy(counter: collections.Counter, choices: int) -> float:
    total = sum(counter.values())
    if total == 0 or choices <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counter.values())
    return entropy / math.log(choices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    raw_cfg = checkpoint["config"]
    cfg = base.Config(**raw_cfg)
    cfg.device = args.device
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    rows, pieces = base.load_rows(cfg, sp)
    test = rows[cfg.train_samples + cfg.valid_samples:]
    loader = DataLoader(
        base.ParallelDataset(test), batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=base.collate(pieces),
    )
    model = LearnedFoldTreeHeapSeq2Seq(pieces + 1, cfg.dim, cfg.hidden).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rng = random.Random(cfg.seed + 911)

    routes_by_length = collections.defaultdict(collections.Counter)
    choices_by_length_step = collections.defaultdict(collections.Counter)
    changed = compared = leftmost = decisions = 0

    with torch.no_grad():
        for src, length, _, _ in loader:
            src, length = src.to(args.device), length.to(args.device)
            model.encode(src, length)
            original = model.last_merge_choices.cpu()
            shuffled = src.clone()
            for row, size in enumerate(length.tolist()):
                values = shuffled[row, :size].tolist()
                rng.shuffle(values)
                shuffled[row, :size] = torch.tensor(values, device=args.device)
            model.encode(shuffled, length)
            perturbed = model.last_merge_choices.cpu()

            for row, size in enumerate(length.cpu().tolist()):
                route = tuple(original[row, :size - 1].tolist())
                other = tuple(perturbed[row, :size - 1].tolist())
                routes_by_length[size][route] += 1
                compared += 1
                changed += int(route != other)
                for step, choice in enumerate(route):
                    choices_by_length_step[(size, step)][choice] += 1
                    leftmost += int(choice == 0)
                    decisions += 1
                if compared >= args.samples:
                    break
            if compared >= args.samples:
                break

    dominant = []
    unique_routes = 0
    for size, counter in sorted(routes_by_length.items()):
        count = sum(counter.values())
        unique_routes += len(counter)
        dominant.append({
            "length": size,
            "samples": count,
            "unique_routes": len(counter),
            "dominant_route_fraction": max(counter.values()) / count,
        })
    entropies = []
    for (size, step), counter in choices_by_length_step.items():
        available = size - step - 1
        if available > 1:
            entropies.append(normalized_entropy(counter, available))

    summary = {
        "checkpoint": args.checkpoint,
        "samples": compared,
        "route_changed_after_token_shuffle": changed / max(1, compared),
        "leftmost_merge_fraction": leftmost / max(1, decisions),
        "mean_normalized_choice_entropy": sum(entropies) / max(1, len(entropies)),
        "unique_routes_total_across_lengths": unique_routes,
        "by_length": dominant,
        "interpretation": "Input-dependent routes are necessary but not sufficient; generation evidence must also show a causal full-vs-leaf/internal-node gain.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
