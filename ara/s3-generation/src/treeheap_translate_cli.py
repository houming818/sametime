#!/usr/bin/env python3
"""Interactive translation CLI for a materialized STONE-2 checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_canonical_codec as c02
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_fixed_root_noise_repair as c08
import s3_wmt_treeheap_seq2seq as base


def load_runtime(checkpoint_path: str, device: str, spm_model: str = ""):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**payload["student_args"])
    args.device = device
    tokens = payload["tokens"]
    bundled = Path(checkpoint_path).resolve().parent.parent / "treeheap_sp.model"
    tokenizer_path = spm_model or (
        str(bundled) if bundled.exists() else args.spm_model
    )
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    model = c02.make_model(
        "canonical_learned", args, tokens["vocab"], tokens["pad"]
    )
    model.decoder = c06.FloorPressureDecoder(
        tokens["vocab"],
        args.dim,
        args.hidden,
        model.encoder.depths,
        c06.DEPTH_FLOOR,
    )
    model = model.to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return payload, args, tokens, sp, model


@torch.no_grad()
def translate(text, args, tokens, sp, model, max_new_tokens):
    ids = sp.encode(text, out_type=int)[: args.heap_width - 1]
    ids.append(tokens["eos"])
    source = torch.tensor([ids], dtype=torch.long, device=args.device)
    length = torch.tensor([len(ids)], dtype=torch.long, device=args.device)
    fixed, visible_length = c08.fixed_source(
        source,
        length,
        "eos_tail",
        args.heap_width,
        tokens["pad"],
        tokens["eos"],
        sp.get_piece_size(),
        args.noise_seed,
    )
    prediction = model.greedy(
        fixed,
        visible_length,
        tokens["bos"],
        tokens["eos"],
        max_new_tokens,
        route_mode="depth_floor",
    )[0].cpu().tolist()
    clean = base.clean(prediction, tokens["eos"], tokens["pad"])
    return sp.decode(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spm-model", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    cli = parser.parse_args()
    payload, args, tokens, sp, model = load_runtime(
        cli.checkpoint, cli.device, cli.spm_model
    )

    if cli.text:
        print(json.dumps({
            "arm": payload["arm"],
            "input": cli.text,
            "output": translate(
                cli.text, args, tokens, sp, model, cli.max_new_tokens
            ),
        }, ensure_ascii=False))
        return

    print(f"TreeHeap STONE-2 CLI ({payload['arm']}); Ctrl-D to exit")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text:
            print(translate(
                text, args, tokens, sp, model, cli.max_new_tokens
            ))


if __name__ == "__main__":
    main()
