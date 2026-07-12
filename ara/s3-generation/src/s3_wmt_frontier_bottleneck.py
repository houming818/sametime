#!/usr/bin/env python3
"""WMT fixed-bandwidth frontier comparison for TreeHeap seq2seq."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base


FRONTIER_K = 4


class AdjacentFrontier(base.Seq2SeqBase):
    def __init__(self, vocab: int, dim: int, hidden: int, route_mode: str):
        super().__init__(vocab, dim, hidden)
        self.route_mode = route_mode
        self.emb = nn.Embedding(vocab, dim)
        self.left = nn.Parameter(torch.randn(dim) * 0.02)
        self.right = nn.Parameter(torch.randn(dim) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Tanh()
        )
        self.merge_score = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        if route_mode != "learned":
            for parameter in self.merge_score.parameters():
                parameter.requires_grad_(False)
        self.last_choices = None
        self.last_active = None

    def _choice(self, scores, valid_pair, masses, src_hash, step):
        batch, choices = scores.shape
        active = valid_pair.any(-1)
        if self.route_mode == "learned":
            masked = scores.masked_fill(~valid_pair, -1e9)
            masked = torch.where(active[:, None], masked, torch.zeros_like(masked))
            if self.training:
                return F.gumbel_softmax(masked, tau=1.0, hard=True, dim=-1)
            return F.one_hot(masked.argmax(-1), num_classes=choices).to(scores.dtype)
        if self.route_mode == "fixed":
            pair_mass = masses[:, :-1] + masses[:, 1:]
            position = torch.arange(choices, device=scores.device, dtype=scores.dtype)[None]
            fixed_score = -pair_mass - position * 1e-4
            fixed_score = fixed_score.masked_fill(~valid_pair, -1e9)
            fixed_score = torch.where(active[:, None], fixed_score, torch.zeros_like(fixed_score))
            return F.one_hot(fixed_score.argmax(-1), num_classes=choices).to(scores.dtype)
        available = valid_pair.sum(-1).clamp_min(1)
        index = ((src_hash + 104729 * (step + 1)) % available).long()
        return F.one_hot(index, num_classes=choices).to(scores.dtype)

    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        batch, width = src.shape
        nodes = self.emb(src)
        masses = (torch.arange(width, device=src.device)[None] < length[:, None]).to(nodes.dtype)
        current_length = length.clone()
        positions = torch.arange(width, device=src.device, dtype=torch.long) + 1
        src_hash = (src.long() * positions[None]).sum(-1).abs()
        choices_log = []
        active_log = []
        step = 0

        while width > FRONTIER_K:
            left, right = nodes[:, :-1], nodes[:, 1:]
            parents = self.compose(torch.cat([left + self.left, right + self.right], dim=-1))
            scores = self.merge_score(torch.cat([left, right, parents], dim=-1)).squeeze(-1)
            active = current_length > FRONTIER_K
            valid_pair = torch.arange(width - 1, device=src.device)[None] < (current_length - 1).clamp_min(0)[:, None]
            valid_pair = valid_pair & active[:, None]
            choice = self._choice(scores, valid_pair, masses, src_hash, step)

            node_candidates = []
            mass_candidates = []
            parent_masses = masses[:, :-1] + masses[:, 1:]
            for pair in range(width - 1):
                node_candidates.append(torch.cat([nodes[:, :pair], parents[:, pair:pair + 1], nodes[:, pair + 2:]], dim=1))
                mass_candidates.append(torch.cat([masses[:, :pair], parent_masses[:, pair:pair + 1], masses[:, pair + 2:]], dim=1))
            proposed_nodes = (choice[:, :, None, None] * torch.stack(node_candidates, dim=1)).sum(1)
            proposed_masses = (choice[:, :, None] * torch.stack(mass_candidates, dim=1)).sum(1)
            nodes = torch.where(active[:, None, None], proposed_nodes, nodes[:, :width - 1])
            masses = torch.where(active[:, None], proposed_masses, masses[:, :width - 1])
            current_length = torch.where(active, current_length - 1, current_length)
            choices_log.append(choice.argmax(-1))
            active_log.append(active)
            width -= 1
            step += 1

        self.last_choices = torch.stack(choices_log, dim=1).detach()
        self.last_active = torch.stack(active_log, dim=1).detach()
        mask = torch.arange(FRONTIER_K, device=src.device)[None] < current_length[:, None]
        return nodes, mask


class LearnedFrontier(AdjacentFrontier):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__(vocab, dim, hidden, "learned")


class FixedFrontier(AdjacentFrontier):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__(vocab, dim, hidden, "fixed")


class RandomFrontier(AdjacentFrontier):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__(vocab, dim, hidden, "random")


class FlatK(base.Seq2SeqBase):
    def __init__(self, vocab: int, dim: int, hidden: int):
        super().__init__(vocab, dim, hidden)
        self.emb = nn.Embedding(vocab, dim)
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Tanh())

    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        batch, width = src.shape
        values = self.emb(src)
        position = torch.arange(width, device=src.device)[None].expand(batch, -1)
        bins = torch.div(position * FRONTIER_K, length[:, None].clamp_min(1), rounding_mode="floor").clamp_max(FRONTIER_K - 1)
        valid = position < length[:, None]
        out = values.new_zeros((batch, FRONTIER_K, values.shape[-1]))
        count = values.new_zeros((batch, FRONTIER_K))
        out.scatter_add_(1, bins[:, :, None].expand_as(values), values * valid[:, :, None])
        count.scatter_add_(1, bins, valid.to(values.dtype))
        out = self.project(out / count.clamp_min(1)[:, :, None])
        return out, count > 0


base.MODELS = {
    "learned_frontier": LearnedFrontier,
    "fixed_frontier": FixedFrontier,
    "random_frontier": RandomFrontier,
    "flat_k": FlatK,
    "leaf_oracle": base.FlatSeq2Seq,
}


if __name__ == "__main__":
    base.main()
