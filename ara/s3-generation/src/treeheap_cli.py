#!/usr/bin/env python3
"""Owner: SameTime S3 generation.
Authors: Houming818 and Codex Review.
Created: 2026-07-21.
Updated: 2026-07-21.
Purpose: Run non-teacher-forced translation with a STONE-1 TreeHeap checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_private_protocol as stone
import s3_wmt_treeheap_seq2seq as base


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(checkpoint_path: Path, tokenizer_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "treeheap-stone1-v1":
        raise ValueError("unsupported checkpoint format")
    expected = checkpoint["tokenizer"]["sha256"]
    observed = file_digest(tokenizer_path)
    if observed != expected:
        raise ValueError(
            f"tokenizer SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    config = checkpoint["model_config"]
    args = argparse.Namespace(
        dim=config["dim"],
        hidden=config["hidden"],
        heap_width=config["heap_width"],
        leaf_cut=config["leaf_cut"],
    )
    model = stone.make_model(
        checkpoint["variant"], args, config["vocab"], config["pad"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return model, tokenizer, config


@torch.inference_mode()
def translate(
    model,
    tokenizer,
    config: dict,
    text: str,
    device: str,
    max_new_tokens: int,
) -> dict:
    pieces = tokenizer.encode(text.strip(), out_type=int)
    if not pieces:
        raise ValueError("input text is empty after tokenization")
    source = pieces + [tokenizer.eos_id()]
    if len(source) > config["heap_width"]:
        raise ValueError(
            f"input uses {len(source)} tokens; checkpoint limit is {config['heap_width']}"
        )
    source_tensor = torch.tensor([source], dtype=torch.long, device=device)
    length = torch.tensor([len(source)], dtype=torch.long, device=device)
    predicted, route = model.greedy(
        source_tensor,
        length,
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        max_new_tokens,
    )
    output_tokens = base.clean(
        predicted[0].detach().cpu().tolist(), tokenizer.eos_id(), config["pad"],
    )
    return {
        "source": text,
        "translation": tokenizer.decode(output_tokens),
        "source_tokens": len(source),
        "output_tokens": len(output_tokens),
        "route_depth_mass": route.detach().cpu().tolist() if route is not None else None,
    }


def emit(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["translation"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="treeheap")
    parser.add_argument("command", choices=("translate",))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    tokenizer = Path(args.tokenizer).expanduser().resolve()
    model, sp, config = load_runtime(checkpoint, tokenizer, args.device)

    if args.text:
        emit(translate(
            model, sp, config, args.text, args.device, args.max_new_tokens,
        ), args.json)
    if args.interactive:
        while True:
            try:
                text = input("en> ").strip()
            except EOFError:
                break
            if not text or text.lower() in {"quit", "exit"}:
                break
            emit(translate(
                model, sp, config, text, args.device, args.max_new_tokens,
            ), args.json)
    if not args.text and not args.interactive:
        parser.error("provide --text or --interactive")


if __name__ == "__main__":
    main()
