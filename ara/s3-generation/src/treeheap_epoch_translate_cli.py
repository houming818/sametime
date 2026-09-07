#!/usr/bin/env python3
"""Read-only translation CLI for E01 epoch-repeat checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_structural_protocol_capacity_ladder_d11 as d11  # noqa: E402
import s3_structural_protocol_full_pipeline_d10 as d10  # noqa: E402


DEPTHS = (5, 6, 7)


def load_runtime(
    checkpoint_path: Path,
    device: str,
    source_checkpoint: str = "",
    warm_start: str = "",
    spm_model: str = "",
):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("claim") != "S3-EPOCH-REPEAT-SCALING-E01":
        raise ValueError(f"unsupported checkpoint claim: {payload.get('claim')}")
    run = payload["run"]
    args = argparse.Namespace(**run["config"])
    args.device = device
    args.spm_model = spm_model or args.spm_model
    source_path = Path(source_checkpoint or args.source_checkpoint)
    warm_path = Path(warm_start or args.warm_start)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    args.pad, args.vocab = pieces, pieces + 3
    source_cpu, _, config, source_hash, _ = d03.load_model(
        source_path, args, sp, args.pad, args.vocab
    )
    model, _ = d11.make_model(source_cpu, config, args, warm_path, payload)
    model.eval()
    return payload, args, sp, model, pieces, eos, bos, source_hash


@torch.inference_mode()
def translate(text: str, direction: str, depth: int, runtime, max_new_tokens: int) -> dict:
    payload, args, sp, model, pieces, eos, bos, _ = runtime
    raw = sp.encode(text.strip(), out_type=int)
    if not raw:
        raise ValueError("input is empty after tokenization")
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    source_ids = [direction_ids[direction], *raw[:32], eos]
    source = torch.tensor([source_ids], dtype=torch.long, device=args.device)
    lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)
    generated, route, budgets, _, _ = model.greedy(
        source, lengths, bos, eos, max_new_tokens, depth
    )
    output_ids = d10.wmt.clean(generated[0].tolist(), eos, pieces)
    return {
        "arm": payload["arm"],
        "step": int(payload["step"]),
        "cursor": int(payload["cursor"]),
        "direction": direction,
        "depth": depth,
        "input": text,
        "input_pieces": len(raw),
        "input_truncated": len(raw) > 32,
        "output": sp.decode(output_ids),
        "output_pieces": len(output_ids),
        "budget": int(budgets[0]),
        "route_depth_mass": [float(value) for value in route.detach().cpu()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="treeheap-epoch-translate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--warm-start", default="")
    parser.add_argument("--spm-model", default="")
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--direction", choices=("en2zh", "zh2en"), required=True)
    parser.add_argument("--depth", choices=("all", "5", "6", "7"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    runtime = load_runtime(
        Path(args.checkpoint), args.device, args.source_checkpoint,
        args.warm_start, args.spm_model,
    )
    depths = DEPTHS if args.depth == "all" else (int(args.depth),)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "arm": runtime[0]["arm"],
        "source_sha256": runtime[-1],
        "results": [
            translate(text, args.direction, depth, runtime, args.max_new_tokens)
            for text in args.text
            for depth in depths
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
