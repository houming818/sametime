#!/usr/bin/env python3
"""Run non-teacher-forced translation with a STONE-1 C10 checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_c10_long_smoke as c10
import s3_wmt_treeheap_seq2seq as base


def load_runtime(checkpoint_path: Path, tokenizer_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("claim") != "S3-STONE1-FULL-CORPUS-LONG-C10":
        raise ValueError("checkpoint is not a STONE-1 C10 checkpoint")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    config = argparse.Namespace(**checkpoint["config"])
    pieces = sp.get_piece_size()
    pad = pieces
    model, floor = c10.make_model(config, pieces + 1, pad)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, sp, config, pad, floor, checkpoint


@torch.inference_mode()
def translate(model, sp, config, pad: int, text: str, max_output: int, device: str):
    source = sp.encode(text, out_type=int) + [sp.eos_id()]
    if len(source) > int(config.source_width):
        raise ValueError(
            f"input uses {len(source)} pieces; C10 source limit is {config.source_width}"
        )

    source_tensor = torch.tensor(source, dtype=torch.long, device=device)[None]
    length = torch.tensor([len(source)], dtype=torch.long, device=device)
    fixed, visible_length = c10.fixed_source(
        source_tensor, length, config, pad, sp.eos_id(), sp.get_piece_size(),
    )
    predicted, route = model.greedy(
        fixed,
        visible_length,
        sp.bos_id(),
        sp.eos_id(),
        max_output,
        route_mode="depth_floor",
    )
    token_ids = base.clean(
        predicted[0].detach().cpu().tolist(), sp.eos_id(), pad,
    )
    return {
        "input": text,
        "translation": sp.decode(token_ids),
        "input_pieces": len(source),
        "output_pieces": len(token_ids),
        "route_mass_by_level": [float(value) for value in route.detach().cpu()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="treeheap-c10")
    parser.add_argument(
        "--checkpoint",
        default=(
            "ara/s3-generation/evidence/s3_stone1_c10_raw_full_train/"
            "checkpoint_latest.pt"
        ),
    )
    parser.add_argument(
        "--spm-model",
        default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--max-output", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model, sp, config, pad, floor, checkpoint = load_runtime(
        Path(args.checkpoint).expanduser().resolve(),
        Path(args.spm_model).expanduser().resolve(),
        args.device,
    )
    texts = list(args.text)
    if not texts:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        raise SystemExit("provide --text or newline-delimited stdin")

    metadata = {
        "claim": checkpoint["claim"],
        "global_step": checkpoint["global_step"],
        "processed_tokens": checkpoint["processed_tokens"],
        "heap_width": config.heap_width,
        "source_width": config.source_width,
        "depth_floor_per_level": floor,
        "device": args.device,
    }
    if args.json:
        print(json.dumps({"model": metadata}, ensure_ascii=False))
    else:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))

    for text in texts:
        try:
            result = translate(
                model, sp, config, pad, text, args.max_output, args.device,
            )
        except ValueError as error:
            result = {"input": text, "error": str(error)}
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
            continue
        print(f"EN: {result['input']}")
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"ZH: {result['translation']}")
            print(
                f"pieces: input={result['input_pieces']} "
                f"output={result['output_pieces']}"
            )
            route = ", ".join(
                f"{value:.4f}" for value in result["route_mass_by_level"]
            )
            print(f"route: [{route}]")
        print()


if __name__ == "__main__":
    main()
