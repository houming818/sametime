#!/usr/bin/env python3
"""CLI inference for S2 adaptive lifting WMT checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base
import s2_adaptive_lifting_wmt as experiment


def load_model(checkpoint_path: str, sp, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    name = checkpoint["name"]
    vocab = sp.get_piece_size() + 1
    pad = sp.get_piece_size()
    model = experiment.make_model(
        name,
        vocab,
        int(config["dim"]),
        int(config["hidden"]),
        int(config["heap_width"]),
        pad,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, name, config, pad


@torch.no_grad()
def translate(model, text: str, sp, config: dict, pad: int, device: str, max_output: int):
    source = sp.encode(text, out_type=int) + [sp.eos_id()]
    heap_width = int(config["heap_width"])
    if len(source) > heap_width:
        raise ValueError(
            f"source has {len(source)} pieces, but checkpoint heap width is {heap_width}"
        )
    src = torch.tensor(source, dtype=torch.long, device=device)[None]
    length = torch.tensor([len(source)], dtype=torch.long, device=device)
    predicted, route = model.greedy(
        src,
        length,
        sp.bos_id(),
        sp.eos_id(),
        max_output,
    )
    pieces = base.clean(predicted[0].detach().cpu().tolist(), sp.eos_id(), pad)
    result = {"translation": sp.decode(pieces), "source_pieces": len(source)}
    if route is not None:
        result["route_depth_mass"] = [float(value) for value in route.detach().cpu()]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--spm-model",
        default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    )
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--max-output", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    model, name, config, pad = load_model(args.checkpoint, sp, args.device)
    texts = list(args.text)
    if not texts:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        raise SystemExit("provide --text or newline-delimited stdin")

    print(f"model={name} checkpoint={args.checkpoint}")
    for text in texts:
        try:
            result = translate(
                model, text, sp, config, pad, args.device, args.max_output,
            )
            print(f"EN: {text}")
            print(f"ZH: {result['translation']}")
            print(f"pieces: {result['source_pieces']}")
            if "route_depth_mass" in result:
                route = ", ".join(f"{value:.4f}" for value in result["route_depth_mass"])
                print(f"route: [{route}]")
        except ValueError as error:
            print(f"EN: {text}")
            print(f"ERROR: {error}")
        print()


if __name__ == "__main__":
    main()
