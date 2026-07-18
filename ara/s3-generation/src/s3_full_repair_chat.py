#!/usr/bin/env python3
"""CLI inference for full-corpus repair-aware TreeHeap checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_full_corpus_repair_seq2seq as full
import s3_wmt_treeheap_seq2seq as base


def load_model(checkpoint_path: str, base_checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    model = full.repair_probe.make_model(base_checkpoint, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate(model, sp, prompt: str, source_width: int, max_new_tokens: int, device: str) -> str:
    pad, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    source_ids = sp.encode(prompt, out_type=int)
    source, length = full.fixed(source_ids, source_width, eos, pad)
    source = source.unsqueeze(0).to(device)
    length = torch.tensor([length], dtype=torch.long, device=device)
    leaf, _, _, fold_masks = model.encoder.fold(source, length)
    memory = leaf + model.resolution.weight[model.depths]
    predicted = model.decoder.greedy(memory, fold_masks[0], bos, eos, max_new_tokens)
    output_ids = base.clean(predicted[0].cpu().tolist(), eos, pad)
    return sp.decode(output_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default="/home/nio/datasets/treeheap_checkpoints/checkpoint_annealed.pt")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    model, checkpoint = load_model(args.checkpoint, args.base_checkpoint, args.device)
    config = checkpoint.get("config", {})
    source_width = int(config.get("source_width", 64))
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "source_width": source_width,
        "device": args.device,
    }, ensure_ascii=False))

    prompts = list(args.prompt)
    if prompts:
        for prompt in prompts:
            print(json.dumps({
                "prompt": prompt,
                "generated": generate(model, sp, prompt, source_width, args.max_new_tokens, args.device),
            }, ensure_ascii=False))
        return

    while True:
        try:
            prompt = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt:
            print(generate(model, sp, prompt, source_width, args.max_new_tokens, args.device))


if __name__ == "__main__":
    main()
