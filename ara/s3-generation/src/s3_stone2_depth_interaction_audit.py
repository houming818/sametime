#!/usr/bin/env python3
"""Frozen coarse/middle/fine subset audit for STONE-2 C03."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


GROUPS = {
    "coarse": frozenset((0, 1, 2)),
    "middle": frozenset((3, 4, 5)),
    "fine": frozenset((6, 7)),
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def subset_key(enabled) -> str:
    return "+".join(sorted(enabled)) if enabled else "none"


def make_group_read(enabled_depths):
    enabled_depths = frozenset(enabled_depths)

    def read(self, hidden, tree, masks, mode="native", ablate_depth=-1):
        del mode, ablate_depth
        base_query = self.query(hidden)
        query = base_query
        frontier = masks[0].to(query.dtype)
        entropies = []
        gain = torch.sigmoid(self.read_gain_logit)
        last_depth = len(tree) - 1
        for depth, (nodes, valid) in enumerate(zip(tree, masks)):
            frontier = frontier * valid.to(frontier.dtype)
            frontier = frontier / frontier.sum(-1, keepdim=True).clamp_min(1e-9)
            local = (frontier[:, :, None] * nodes).sum(1)
            if depth in enabled_depths:
                depth_state = self.depth_embedding.weight[depth][None].expand_as(local)
                query = query + gain * self.read_kernel(query, local, depth_state)
            entropies.append(
                -(frontier.clamp_min(1e-12) * frontier.clamp_min(1e-12).log()).sum(-1).mean()
            )
            if depth == last_depth:
                break
            children = tree[depth + 1].reshape(
                nodes.shape[0], nodes.shape[1], 2, nodes.shape[2],
            )
            child_valid = masks[depth + 1].reshape(nodes.shape[0], nodes.shape[1], 2)
            branch_query = (self.branch(hidden) + gain * (query - base_query))[:, None, None]
            scores = (branch_query * children).sum(-1) / math.sqrt(nodes.shape[-1])
            scores = scores.masked_fill(~child_valid, -1e9)
            probability = F.softmax(scores, dim=-1)
            probability = probability * child_valid.to(probability.dtype)
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-9)
            frontier = (frontier[:, :, None] * probability).reshape(nodes.shape[0], -1)
        return local + (query - base_query), torch.stack(entropies)

    return read


def shapley(values, players):
    result = {}
    factorial = math.factorial
    n = len(players)
    for player in players:
        others = [item for item in players if item != player]
        total = 0.0
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                left = frozenset(subset)
                right = left | {player}
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                total += weight * (values[right] - values[left])
        result[player] = total
    return result


def mobius(values, subset):
    total = 0.0
    members = sorted(subset)
    for size in range(len(members) + 1):
        for child in itertools.combinations(members, size):
            total += (-1) ** (len(members) - size) * values[frozenset(child)]
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-rows", type=int, default=256)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.task_eval_rows = args.eval_rows
    sp = spm.SentencePieceProcessor(model_file=config.spm_model)
    pieces = sp.get_piece_size()
    pad, bos, eos = pieces, sp.bos_id(), sp.eos_id()
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    model = integrated.build_integrated_model(config, pieces + 3, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(args.device)
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    rows = test_rows[:args.eval_rows]

    players = tuple(GROUPS)
    measured = {}
    original_read = model.decoder.read
    for size in range(len(players) + 1):
        for enabled in itertools.combinations(players, size):
            enabled_set = frozenset(enabled)
            enabled_depths = frozenset().union(*(GROUPS[name] for name in enabled_set)) if enabled else frozenset()
            model.decoder.read = MethodType(make_group_read(enabled_depths), model.decoder)
            score = c11.evaluate(model, rows, pad, bos, args.device, config.eval_batch)
            measured[enabled_set] = score["nll"]
    model.decoder.read = original_read

    none_nll = measured[frozenset()]
    values = {subset: none_nll - nll for subset, nll in measured.items()}
    phi = shapley(values, players)
    interactions = {}
    for size in (2, 3):
        for subset in itertools.combinations(players, size):
            frozen = frozenset(subset)
            interactions["+".join(subset)] = mobius(values, frozen)

    repeat = []
    for enabled in (frozenset(), frozenset(players)):
        enabled_depths = frozenset().union(*(GROUPS[name] for name in enabled)) if enabled else frozenset()
        model.decoder.read = MethodType(make_group_read(enabled_depths), model.decoder)
        repeat.append(c11.evaluate(model, rows, pad, bos, args.device, config.eval_batch)["nll"])
    model.decoder.read = original_read

    all_set = frozenset(players)
    result = {
        "diagnostic": "STONE-2-C03-D01",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rows": len(rows),
        "groups": {name: sorted(depths) for name, depths in GROUPS.items()},
        "nll": {subset_key(subset): measured[subset] for subset in measured},
        "value_over_none": {subset_key(subset): values[subset] for subset in values},
        "shapley": phi,
        "shapley_sum": sum(phi.values()),
        "full_value": values[all_set],
        "interactions": interactions,
        "repeat": {
            "none": repeat[0],
            "all": repeat[1],
            "max_abs_delta": max(abs(repeat[0] - measured[frozenset()]), abs(repeat[1] - measured[all_set])),
        },
        "interpretation": {
            # Historical field name retained so the committed program exactly
            # reproduces task 293. The logic note corrects the interpretation:
            # positive distributed contribution with negative interactions.
            "distributed_synergy_candidate": values[all_set] > 0.0 and sum(value > 0.0 for value in phi.values()) >= 2,
            "single_group_dominant": sum(value > 0.0 for value in phi.values()) == 1,
            "deterministic": max(abs(repeat[0] - measured[frozenset()]), abs(repeat[1] - measured[all_set])) < 1e-9,
            "formal_training_authorized": False,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
