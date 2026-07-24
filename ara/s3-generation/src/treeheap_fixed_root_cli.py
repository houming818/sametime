#!/usr/bin/env python3
"""Run C08 fixed-root EOS-frame translation without teacher forcing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_fixed_root_noise_repair as c08
import s3_wmt_treeheap_seq2seq as base


def load_runtime(args):
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    pieces = tokenizer.get_piece_size()
    pad, vocab = pieces, pieces + 1
    model_args = argparse.Namespace(
        c04_checkpoint=args.encoder_checkpoint,
        dim=args.dim,
        hidden=args.hidden,
        heap_width=args.heap_width,
        leaf_cut=args.leaf_cut,
    )
    model = c06.load_model(model_args, vocab, pad)
    decoder = torch.load(
        args.decoder_checkpoint, map_location="cpu", weights_only=False,
    )
    if decoder.get("arm") != "eos_tail":
        raise ValueError("C08 CLI requires the eos_tail decoder arm")
    model.decoder.load_state_dict(decoder["decoder_state_dict"], strict=True)
    model.to(args.device).eval()
    return model, tokenizer, pieces, pad


@torch.inference_mode()
def translate(model, tokenizer, pieces, pad, text, args):
    tokens = tokenizer.encode(text.strip(), out_type=int)
    if not tokens:
        raise ValueError("input is empty after tokenization")
    tokens.append(tokenizer.eos_id())
    if len(tokens) > args.heap_width:
        raise ValueError(
            f"input uses {len(tokens)} tokens; limit is {args.heap_width}"
        )
    source = torch.tensor([tokens], dtype=torch.long, device=args.device)
    length = torch.tensor([len(tokens)], dtype=torch.long, device=args.device)
    source, visible_length = c08.fixed_source(
        source, length, "eos_tail", args.heap_width, pad,
        tokenizer.eos_id(), pieces, args.noise_seed,
    )
    predicted, route = model.greedy(
        source, visible_length, tokenizer.bos_id(), tokenizer.eos_id(),
        args.max_new_tokens, route_mode="depth_floor",
    )
    output_tokens = base.clean(
        predicted[0].cpu().tolist(), tokenizer.eos_id(), pad,
    )
    return {
        "source": text,
        "translation": tokenizer.decode(output_tokens),
        "source_tokens": len(tokens),
        "physical_leaves": args.heap_width,
        "tail_fill": "eos",
        "route_depth_mass": route.detach().cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(prog="treeheap-fixed-root")
    parser.add_argument("command", choices=("translate",))
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--noise-seed", type=int, default=74231)
    args = parser.parse_args()
    model, tokenizer, pieces, pad = load_runtime(args)

    def emit(text):
        result = translate(model, tokenizer, pieces, pad, text, args)
        print(
            json.dumps(result, ensure_ascii=False)
            if args.json else result["translation"],
            flush=True,
        )

    for text in args.text:
        emit(text)
    if args.interactive:
        while True:
            try:
                text = input("en> ").strip()
            except EOFError:
                break
            if not text or text.lower() in {"quit", "exit"}:
                break
            emit(text)
    if not args.text and not args.interactive:
        parser.error("provide --text or --interactive")


if __name__ == "__main__":
    main()
