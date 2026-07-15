#!/usr/bin/env python3
"""Human-readable microscope for S3-TREEHEAP-GEOMETRY-C01 states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decode(sp, ids: torch.Tensor) -> str:
    return sp.decode([int(x) for x in ids.tolist()]).replace("\n", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--model-script", default=str(here / "s3_treeheap_multiscale_geometry.py"))
    parser.add_argument("--base-script", default=str(here / "s3_residual_treeheap_forest_pretrain.py"))
    parser.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_treeheap_multiscale_geometry_smoke/checkpoint.pt")
    parser.add_argument("--block-dir", default="/home/nio/datasets/derived/s3_residual_treeheap_forest/full_blocks64")
    parser.add_argument("--output", default="ara/s3-generation/evidence/s3_treeheap_multiscale_geometry_smoke/layer_observation.json")
    parser.add_argument("--pool-blocks", type=int, default=512)
    parser.add_argument("--focus-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71501)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    geometry = load_module("geometry_probe", Path(args.model_script))
    base = load_module("geometry_data", Path(args.base_script))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    manifest = base.manifest(Path(args.block_dir), "valid")
    vocab = int(manifest["tokenizer"]["vocab"])
    model = geometry.MultiscaleTreeHeap(vocab, config["dim"], config["sketch_dim"], config["depths"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sketch = geometry.ExactSketches(vocab, config["sketch_dim"], config["seed"] + 100, device)
    tokenizer_path = Path(manifest["tokenizer"]["path"])
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    token_batches = []
    for tokens, _ in geometry.batches(base, Path(args.block_dir), "valid", 64, args.seed + 2, args.pool_blocks):
        token_batches.append(tokens)
    tokens = torch.cat(token_batches)[: args.pool_blocks].to(device)
    query = tokens[:1]
    candidates = tokens[1:]

    with torch.no_grad():
        query_states = model.levels(query)
        query_bags, query_adjs = sketch.levels(query)
        candidate_states = model.levels(candidates)
        candidate_bags, candidate_adjs = sketch.levels(candidates)

    rows = []
    with torch.no_grad():
        for depth_index in range(config["depths"]):
            depth = depth_index + 1
            width = 2 ** depth
            q_state = query_states[depth_index][0, 0]
            q_bag = query_bags[depth_index][0, 0]
            q_adj = query_adjs[depth_index][0, 0]
            pool_state = candidate_states[depth_index].flatten(0, 1)
            pool_bag = candidate_bags[depth_index].flatten(0, 1)
            pool_adj = candidate_adjs[depth_index].flatten(0, 1)
            scores = F.cosine_similarity(q_state[None], pool_state, dim=-1)
            nearest_flat = int(scores.argmax())
            nodes_per_block = candidate_states[depth_index].shape[1]
            block_index = nearest_flat // nodes_per_block
            node_index = nearest_flat % nodes_per_block
            start = node_index * width
            nearest_tokens = candidates[block_index, start : start + width]
            bag_pred = model.bag_read(q_state)
            adj_pred = model.adj_read(q_state)
            rows.append({
                "depth": depth,
                "covered_tokens": width,
                "query_text": decode(sp, query[0, :width].cpu()),
                "nearest_text": decode(sp, nearest_tokens.cpu()),
                "state_norm": float(q_state.norm()),
                "state_first_8": [round(float(x), 4) for x in q_state[:8]],
                "own_bag_read_cosine": float(F.cosine_similarity(bag_pred, q_bag, dim=0)),
                "own_adjacency_read_cosine": float(F.cosine_similarity(adj_pred, q_adj, dim=0)),
                "nearest_state_cosine": float(scores[nearest_flat]),
                "nearest_exact_bag_cosine": float(F.cosine_similarity(q_bag, pool_bag[nearest_flat], dim=0)),
                "nearest_exact_adjacency_cosine": float(F.cosine_similarity(q_adj, pool_adj[nearest_flat], dim=0)),
            })

        focus_index = args.focus_depth - 1
        focus_width = 2 ** args.focus_depth
        focus_states = query_states[focus_index][0]
        focus_bags = query_bags[focus_index][0]
        focus_adjs = query_adjs[focus_index][0]
        focus_nodes = []
        for node_index, (state, bag, adj) in enumerate(zip(focus_states, focus_bags, focus_adjs)):
            start = node_index * focus_width
            focus_nodes.append({
                "node": node_index,
                "token_range": [start, start + focus_width],
                "text": decode(sp, query[0, start : start + focus_width].cpu()),
                "bag_read_cosine": float(F.cosine_similarity(model.bag_read(state), bag, dim=0)),
                "adjacency_read_cosine": float(F.cosine_similarity(model.adj_read(state), adj, dim=0)),
                "state_first_8": [round(float(x), 4) for x in state[:8]],
            })
        focus_similarity = F.normalize(focus_states, dim=-1) @ F.normalize(focus_states, dim=-1).T

    result = {
        "description": "One nested left-edge node per depth; nearest neighbours exclude the query block.",
        "query_full_text": decode(sp, query[0].cpu()),
        "pool_blocks": int(candidates.shape[0]),
        "rows": rows,
        "focus_layer": {
            "depth": args.focus_depth,
            "covered_tokens_per_node": focus_width,
            "nodes": focus_nodes,
            "within_layer_state_cosine": [[round(float(x), 4) for x in row] for row in focus_similarity],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
