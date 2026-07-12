#!/usr/bin/env python3
"""WMT seq2seq with translation-loss-selected TreeHeap adjacent folds."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_wmt_treeheap_seq2seq as base


class LearnedFoldTreeHeapSeq2Seq(base.Seq2SeqBase):
    """A hard-forward, soft-backward adjacent-fold TreeHeap encoder.

    At every reduction step all adjacent pairs propose a parent.  A learned
    kernel selects one pair with straight-through Gumbel-Softmax during
    training and argmax during evaluation.  Repeating this operation produces
    one explicit binary merge history for each source sentence.
    """

    def __init__(self, vocab: int, dim: int, hidden: int, tau: float = 1.0):
        super().__init__(vocab, dim, hidden)
        self.emb = nn.Embedding(vocab, dim)
        self.left = nn.Parameter(torch.randn(dim) * 0.02)
        self.right = nn.Parameter(torch.randn(dim) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Tanh()
        )
        self.merge_score = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.tau = tau
        self.last_merge_choices: torch.Tensor | None = None
        self.last_merge_active: torch.Tensor | None = None

    def encode(self, src: torch.Tensor, length: torch.Tensor, mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
        batch, width = src.shape
        leaf_mask = torch.arange(width, device=src.device)[None] < length[:, None]
        nodes = self.emb(src)
        leaves = nodes
        current_length = length.clone()
        merged_states: List[torch.Tensor] = []
        merged_masks: List[torch.Tensor] = []
        merge_choices: List[torch.Tensor] = []

        while width > 1:
            left = nodes[:, :-1]
            right = nodes[:, 1:]
            parents = self.compose(torch.cat([left + self.left, right + self.right], dim=-1))
            scores = self.merge_score(torch.cat([left, right, parents], dim=-1)).squeeze(-1)
            active = current_length > 1
            valid_pair = torch.arange(width - 1, device=src.device)[None] < (current_length - 1).clamp_min(0)[:, None]
            scores = scores.masked_fill(~valid_pair, -1e9)
            scores = torch.where(active[:, None], scores, torch.zeros_like(scores))

            if self.training:
                choice = F.gumbel_softmax(scores, tau=self.tau, hard=True, dim=-1)
            else:
                choice = F.one_hot(scores.argmax(-1), num_classes=width - 1).to(scores.dtype)

            candidate_sequences = []
            for pair in range(width - 1):
                candidate_sequences.append(
                    torch.cat([nodes[:, :pair], parents[:, pair:pair + 1], nodes[:, pair + 2:]], dim=1)
                )
            candidates = torch.stack(candidate_sequences, dim=1)
            reduced = (choice[:, :, None, None] * candidates).sum(dim=1)
            fallback = nodes[:, :width - 1]
            nodes = torch.where(active[:, None, None], reduced, fallback)

            selected_parent = (choice[:, :, None] * parents).sum(dim=1)
            merged_states.append(selected_parent)
            merged_masks.append(active)
            merge_choices.append(choice.argmax(-1))
            current_length = (current_length - 1).clamp_min(1)
            width -= 1

        internal = torch.stack(merged_states, dim=1)
        internal_mask = torch.stack(merged_masks, dim=1)
        choices = torch.stack(merge_choices, dim=1)
        self.last_merge_choices = choices.detach()
        self.last_merge_active = internal_mask.detach()

        if mode == "leaf_only":
            return leaves, leaf_mask
        if mode == "internal_only":
            return internal, internal_mask
        if mode == "root_only":
            root_index = (length - 2).clamp_min(0)
            root = internal[torch.arange(batch, device=src.device), root_index]
            root = torch.where((length > 1)[:, None], root, leaves[:, 0])
            return root[:, None], torch.ones((batch, 1), device=src.device, dtype=torch.bool)
        return torch.cat([leaves, internal], dim=1), torch.cat([leaf_mask, internal_mask], dim=1)


base.MODELS = {
    "treeheap": LearnedFoldTreeHeapSeq2Seq,
    "fixed_tree": base.TreeHeapSeq2Seq,
    "flat_seq": base.FlatSeq2Seq,
    "bow": base.BowSeq2Seq,
}


if __name__ == "__main__":
    base.main()
