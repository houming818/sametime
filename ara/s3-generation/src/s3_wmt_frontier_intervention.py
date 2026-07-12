#!/usr/bin/env python3
"""Causal route replacement audit for a trained WMT frontier checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
from s3_wmt_frontier_bottleneck import LearnedFrontier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = base.Config(**checkpoint["config"])
    cfg.device = args.device
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    rows, pieces = base.load_rows(cfg, sp)
    test = rows[cfg.train_samples + cfg.valid_samples:]

    def loader():
        return DataLoader(base.ParallelDataset(test), batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=base.collate(pieces))

    model = LearnedFrontier(pieces + 1, cfg.dim, cfg.hidden).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metrics = {}
    for route_mode in ("learned", "fixed", "random"):
        model.route_mode = route_mode
        result = base.evaluate(model, loader(), cfg, pieces, sp.bos_id(), sp.eos_id(), sp)
        metrics[route_mode] = {key: value for key, value in result.items() if key != "examples"}

    learned_fixed_same = learned_random_same = decisions = 0
    with torch.no_grad():
        for src, length, _, _ in loader():
            src, length = src.to(args.device), length.to(args.device)
            model.route_mode = "learned"
            model.encode(src, length)
            learned, active = model.last_choices.clone(), model.last_active.clone()
            model.route_mode = "fixed"
            model.encode(src, length)
            fixed = model.last_choices.clone()
            model.route_mode = "random"
            model.encode(src, length)
            random = model.last_choices.clone()
            decisions += int(active.sum().item())
            learned_fixed_same += int(((learned == fixed) & active).sum().item())
            learned_random_same += int(((learned == random) & active).sum().item())

    summary = {
        "claim": "S3-WMT-FRONTIER-C01",
        "same_checkpoint_route_intervention": metrics,
        "route_decisions": decisions,
        "learned_fixed_choice_agreement": learned_fixed_same / max(1, decisions),
        "learned_random_choice_agreement": learned_random_same / max(1, decisions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
