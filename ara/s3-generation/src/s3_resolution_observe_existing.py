#!/usr/bin/env python3
"""Observe an existing C08 checkpoint at each visible TreeHeap depth."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_fixed_root_noise_repair as c08
import s3_wmt_treeheap_seq2seq as base
import treeheap_fixed_root_cli as cli


DEFAULT_TEXTS = [
    "The earth is round.",
    "The apple is sweet.",
    "Why is the window wet?",
    "A cat is eating some food.",
    "I arrived home at seven o'clock.",
]


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def encode_source(model, tokenizer, pieces: int, pad: int, text: str, args):
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
    state = model.states(source, visible_length)
    return tokens, state


@torch.inference_mode()
def first_token_observation(model, levels, masks, tokenizer, pieces: int, args):
    decoder = model.decoder
    hidden = levels[0].new_zeros((1, decoder.hidden))
    previous = torch.tensor(
        [tokenizer.bos_id()], dtype=torch.long, device=args.device,
    )
    context, route = decoder.read(hidden, levels, masks, "depth_floor")
    hidden = decoder.cell(
        torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
    )
    logits = decoder.output(torch.cat((hidden, context), dim=-1))
    probability = F.softmax(logits.float(), dim=-1)[0]
    token_probability = probability[:pieces]
    entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum())
    values, indices = token_probability.topk(args.top_k)
    return {
        "entropy_nats": entropy,
        "effective_candidates_exp_entropy": math.exp(min(20.0, entropy)),
        "top_k_mass": float(values.sum()),
        "top_tokens": [
            {
                "id": int(index),
                "piece": tokenizer.id_to_piece(int(index)),
                "decoded": tokenizer.decode([int(index)]),
                "probability": float(value),
            }
            for value, index in zip(values, indices)
        ],
        "route_mass": route.detach().float().cpu().tolist(),
    }


@torch.inference_mode()
def observe_text(model, tokenizer, pieces: int, pad: int, text: str, args):
    tokens, state = encode_source(model, tokenizer, pieces, pad, text, args)
    all_levels, all_masks = model.visible(state[3], state[4])
    root_hash = tensor_digest(all_levels[0])
    observations = []
    for visible_count in range(1, len(all_levels) + 1):
        levels = all_levels[:visible_count]
        masks = all_masks[:visible_count]
        first = first_token_observation(
            model, levels, masks, tokenizer, pieces, args,
        )
        predicted, route = model.decoder.greedy(
            levels, masks, tokenizer.bos_id(), tokenizer.eos_id(),
            args.max_new_tokens, route_mode="depth_floor",
        )
        output_tokens = base.clean(
            predicted[0].cpu().tolist(), tokenizer.eos_id(), pad,
        )
        observations.append({
            "visible_level_count": visible_count,
            "deepest_visible_depth": visible_count - 1,
            "node_count_at_deepest_level": int(levels[-1].shape[1]),
            "first_token": first,
            "generation": tokenizer.decode(output_tokens),
            "generation_token_count": len(output_tokens),
            "mean_generation_route_mass": route.detach().float().cpu().tolist(),
        })
    return {
        "source": text,
        "source_piece_count_before_fixed_tail": len(tokens),
        "physical_leaves": args.heap_width,
        "same_h_state_root_sha256": root_hash,
        "available_visible_levels": len(all_levels),
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--noise-seed", type=int, default=74231)
    args = parser.parse_args()

    model, tokenizer, pieces, pad = cli.load_runtime(args)
    rows = [
        observe_text(model, tokenizer, pieces, pad, text, args)
        for text in (args.text or DEFAULT_TEXTS)
    ]
    result = {
        "kind": "existing_checkpoint_resolution_observation",
        "training": False,
        "claim_registered": False,
        "checkpoint": {
            "encoder": args.encoder_checkpoint,
            "decoder": args.decoder_checkpoint,
        },
        "device": args.device,
        "rows": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
