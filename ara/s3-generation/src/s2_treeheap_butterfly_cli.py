#!/usr/bin/env python3
"""Inspect a trained TreeHeap Butterfly checkpoint on custom text."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from s2_treeheap_butterfly_wmt import ButterflyRecursive  # noqa: E402


def clean(ids: list[int], eos: int, pad: int) -> list[int]:
    result = []
    for token in ids:
        if token in (eos, pad):
            break
        result.append(token)
    return result


def load_model(checkpoint_path: Path, sp, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    pieces = sp.get_piece_size()
    pad = pieces
    model = ButterflyRecursive(
        vocab=pieces + 1,
        dim=int(cfg["dim"]),
        hidden=int(cfg["hidden"]),
        heap_width=int(cfg["heap_width"]),
        pad=pad,
        mode="butterfly",
        scale=float(cfg["coupling_scale"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, pad


@torch.no_grad()
def inspect_mode(model, src, length, sp, pad, mode: str, max_output: int, reference: str | None):
    bos, eos = sp.bos_id(), sp.eos_id()
    model.encoder.runtime_mode = mode
    generated, route = model.greedy(src, length, bos, eos, max_output)
    generated_ids = clean(generated[0].tolist(), eos, pad)
    result = {
        "mode": mode,
        "generation_zh": sp.decode(generated_ids),
        "output_pieces": len(generated_ids),
        "route_depth_mass": [round(float(value), 6) for value in route.cpu()],
    }
    if reference is not None:
        target_ids = sp.encode(reference, out_type=int) + [eos]
        target = torch.tensor([target_ids], dtype=torch.long, device=src.device)
        logits, _ = model.teacher(src, length, target, bos)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="mean",
        )
        result["reference_zh"] = reference
        result["reference_nll"] = float(loss)
        result["reference_ppl"] = math.exp(min(20.0, float(loss)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native Butterfly communication with runtime topology ablations.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument("--text", required=True, help="English source text")
    parser.add_argument("--reference-zh", help="Optional Chinese reference for per-example NLL")
    parser.add_argument("--mode", choices=("all", "butterfly", "identity", "adjacent"), default="all")
    parser.add_argument("--max-output", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    model, checkpoint, pad = load_model(Path(args.checkpoint), sp, args.device)
    eos = sp.eos_id()
    source_ids = sp.encode(args.text, out_type=int) + [eos]
    heap_width = model.encoder.heap_width
    if len(source_ids) > heap_width:
        raise SystemExit(f"source has {len(source_ids)} pieces but heap width is {heap_width}")

    warning = None
    training_min = int(checkpoint["config"].get("min_len", 0))
    training_max = int(checkpoint["config"].get("max_len", heap_width - 1))
    raw_length = len(source_ids) - 1
    if not training_min <= raw_length <= training_max:
        warning = (
            f"input has {raw_length} raw pieces; training range was "
            f"{training_min}..{training_max}, so this is out of distribution"
        )

    src = torch.tensor([source_ids], dtype=torch.long, device=args.device)
    length = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)
    modes = ("butterfly", "identity", "adjacent") if args.mode == "all" else (args.mode,)
    results = [
        inspect_mode(
            model, src, length, sp, pad, mode, args.max_output, args.reference_zh,
        )
        for mode in modes
    ]
    model.encoder.runtime_mode = None

    report = {
        "claim": checkpoint.get("claim"),
        "seed": checkpoint.get("seed"),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_en": args.text,
        "source_raw_pieces": raw_length,
        "warning": warning,
        "interpretation": (
            "Lower reference_nll is better. The preregistered causal test predicts "
            "native butterfly should usually beat runtime identity/adjacent overrides "
            "in aggregate, not necessarily on every single sentence."
        ),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
