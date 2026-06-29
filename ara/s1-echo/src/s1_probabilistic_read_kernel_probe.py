#!/usr/bin/env python3
"""S1 probabilistic TreeHeap read kernel probe.

SPR-032 tests a refined read hypothesis:

    TreeHeap read is not leaf lookup and not root-only decoding.
    It is query-conditioned probabilistic collapse over stop/left/right.

The experiment uses real WMT English SentencePiece token sequences. We write a
short sequence into the leaves of a complete binary heap. A query asks for a
target heap node. If the target is a leaf, the output is the token id at that
leaf. If the target is an internal node, the output is a deterministic checksum
bucket of the subheap span. Therefore internal `stop` has meaning.

Models:
    root_query_decoder:
        root state + query id -> label. This is the root bottleneck baseline.

    probabilistic_read_kernel:
        repeatedly applies K_read(q, h_node, path_prefix) -> stop/left/right.
        It has both a hard tail-recursive interpreter and a soft frontier
        interpreter.

This is not translation and not semantic world modeling. It is a read-side
collapse mechanism proof.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PAD = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_wmt_english(path: Path, limit: int) -> Iterable[str]:
    seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if seen >= limit:
                break
            line = line.strip()
            if not line:
                continue
            text = line.split("\t", 1)[0].strip()
            if text:
                seen += 1
                yield text


def collect_sequences(
    wmt_path: Path,
    spm_model: Path,
    samples: int,
    min_len: int,
    max_len: int,
    vocab_limit: int,
    scan_lines: int,
) -> Tuple[List[List[int]], List[str]]:
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model))
    seqs: List[List[int]] = []
    examples: List[str] = []
    for text in read_wmt_english(wmt_path, scan_lines):
        ids = [int(i) + 1 for i in sp.encode(text, out_type=int)]
        ids = [i for i in ids if 0 < i < vocab_limit]
        if min_len <= len(ids) <= max_len:
            seqs.append(ids)
            if len(examples) < 5:
                examples.append(text)
            if len(seqs) >= samples:
                break
    if len(seqs) < samples:
        raise RuntimeError(f"collected {len(seqs)} sequences, need {samples}")
    return seqs, examples


def node_span(node: int, max_len: int) -> Tuple[int, int]:
    """Return [start, end) leaf span for 1-based heap node."""
    start = node
    end = node
    while start < max_len:
        start *= 2
        end = end * 2 + 1
    start_leaf = start - max_len
    end_leaf = end - max_len + 1
    return max(0, start_leaf), min(max_len, end_leaf)


def path_actions(node: int) -> List[int]:
    """Teacher actions from root to node, then stop.

    action ids: stop=0, left=1, right=2
    """
    bits: List[int] = []
    cur = node
    while cur > 1:
        bits.append(1 if cur % 2 == 0 else 2)
        cur //= 2
    return list(reversed(bits)) + [0]


def tail_recursive_interpreter(node: int, max_steps: int = 16) -> int:
    """Deterministic sanity check: follow teacher path with a while loop."""
    actions = path_actions(node)
    cur = 1
    for step, action in enumerate(actions):
        if step >= max_steps:
            raise RuntimeError("tail recursion did not terminate")
        if action == 0:
            return cur
        if action == 1:
            cur = cur * 2
        elif action == 2:
            cur = cur * 2 + 1
    raise RuntimeError("missing stop")


def checksum_label(tokens: List[int], start: int, end: int, vocab_limit: int, buckets: int) -> int:
    vals = tokens[start:end]
    vals = [x for x in vals if x != PAD]
    if not vals:
        return PAD
    acc = 0
    for i, val in enumerate(vals):
        acc = (acc + (i + 1) * int(val)) % buckets
    return vocab_limit + acc


@dataclass
class Item:
    tokens: torch.Tensor
    query_node: torch.Tensor
    target: torch.Tensor
    teacher_actions: torch.Tensor
    teacher_mask: torch.Tensor
    is_internal: torch.Tensor
    depth: torch.Tensor


class NodeQueryDataset(Dataset):
    def __init__(
        self,
        seqs: List[List[int]],
        max_len: int,
        vocab_limit: int,
        checksum_buckets: int,
        train_mode: bool,
        seed: int,
    ) -> None:
        self.seqs = seqs
        self.max_len = max_len
        self.vocab_limit = vocab_limit
        self.checksum_buckets = checksum_buckets
        self.train_mode = train_mode
        self.rng = random.Random(seed)
        self.max_depth = int(np.log2(max_len)) + 1
        self.nodes = list(range(1, max_len * 2))
        self.eval_index: List[Tuple[int, int]] = []
        if not train_mode:
            for i in range(len(seqs)):
                padded = self._pad(seqs[i])
                for node in self.valid_nodes(padded):
                    self.eval_index.append((i, node))

    def _pad(self, ids: List[int]) -> List[int]:
        return ids + [PAD] * (self.max_len - len(ids))

    def valid_nodes(self, padded: List[int]) -> List[int]:
        valid = []
        for node in self.nodes:
            s, e = node_span(node, self.max_len)
            if any(x != PAD for x in padded[s:e]):
                valid.append(node)
        return valid

    def __len__(self) -> int:
        return len(self.seqs) if self.train_mode else len(self.eval_index)

    def make_item(self, seq: List[int], node: int) -> Item:
        padded = self._pad(seq)
        s, e = node_span(node, self.max_len)
        is_leaf = node >= self.max_len
        if is_leaf:
            target = padded[node - self.max_len]
        else:
            target = checksum_label(padded, s, e, self.vocab_limit, self.checksum_buckets)

        actions = path_actions(node)
        padded_actions = actions + [-1] * (self.max_depth - len(actions))
        teacher_mask = [1] * len(actions) + [0] * (self.max_depth - len(actions))
        depth = len(actions) - 1
        return Item(
            tokens=torch.tensor(padded, dtype=torch.long),
            query_node=torch.tensor(node, dtype=torch.long),
            target=torch.tensor(target, dtype=torch.long),
            teacher_actions=torch.tensor(padded_actions, dtype=torch.long),
            teacher_mask=torch.tensor(teacher_mask, dtype=torch.bool),
            is_internal=torch.tensor(not is_leaf, dtype=torch.bool),
            depth=torch.tensor(depth, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> Item:
        if self.train_mode:
            seq = self.seqs[idx]
            padded = self._pad(seq)
            node = self.rng.choice(self.valid_nodes(padded))
        else:
            seq_idx, node = self.eval_index[idx]
            seq = self.seqs[seq_idx]
        return self.make_item(seq, node)


def collate(items: List[Item]) -> Dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([x.tokens for x in items]),
        "query_node": torch.stack([x.query_node for x in items]),
        "target": torch.stack([x.target for x in items]),
        "teacher_actions": torch.stack([x.teacher_actions for x in items]),
        "teacher_mask": torch.stack([x.teacher_mask for x in items]),
        "is_internal": torch.stack([x.is_internal for x in items]),
        "depth": torch.stack([x.depth for x in items]),
    }


class Compose(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class TreeEncoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, dim: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.nodes = max_len * 2 - 1
        self.start = max_len - 1
        self.token_emb = nn.Embedding(vocab, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.compose = Compose(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        device = tokens.device
        states = torch.zeros(bsz, self.nodes + 1, self.token_emb.embedding_dim, device=device)
        pos = torch.arange(self.max_len, device=device)
        leaves = self.token_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for j in range(self.max_len):
            states[:, self.max_len + j, :] = leaves[:, j, :]
        for node in range(self.max_len - 1, 0, -1):
            left = states[:, node * 2, :]
            right = states[:, node * 2 + 1, :]
            states[:, node, :] = self.compose(left, right)
        return states


class RootQueryDecoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, dim: int, labels: int) -> None:
        super().__init__()
        self.encoder = TreeEncoder(vocab, max_len, dim)
        self.query_emb = nn.Embedding(max_len * 2, dim)
        self.readout = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, labels),
        )

    def forward(self, tokens: torch.Tensor, query_node: torch.Tensor) -> torch.Tensor:
        states = self.encoder(tokens)
        root = states[:, 1, :]
        q = self.query_emb(query_node)
        return self.readout(torch.cat([root, q], dim=-1))


class ProbabilisticReadKernel(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_len: int,
        dim: int,
        labels: int,
        max_depth: int,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.nodes = max_len * 2 - 1
        self.max_depth = max_depth
        self.encoder = TreeEncoder(vocab, max_len, dim)
        self.query_emb = nn.Embedding(max_len * 2, dim)
        self.path_emb = nn.Embedding(max_len * 2, dim)
        self.action = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, 3),
        )
        self.readout = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, labels),
        )

    def action_logits_for_node(
        self,
        states: torch.Tensor,
        q: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> torch.Tensor:
        h = states[torch.arange(states.shape[0], device=states.device), node_ids, :]
        p = self.path_emb(node_ids)
        return self.action(torch.cat([q, h, p], dim=-1))

    def teacher_forced_state(
        self,
        states: torch.Tensor,
        query_node: torch.Tensor,
        teacher_actions: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = states.shape[0]
        device = states.device
        q = self.query_emb(query_node)
        node = torch.ones(bsz, dtype=torch.long, device=device)
        action_losses = []
        final = states[:, 1, :]
        for t in range(self.max_depth):
            logits = self.action_logits_for_node(states, q, node)
            mask_t = teacher_mask[:, t]
            if mask_t.any():
                action_losses.append(
                    F.cross_entropy(logits[mask_t], teacher_actions[:, t][mask_t])
                )
            act = teacher_actions[:, t].clamp_min(0)
            stop = act == 0
            final = torch.where(stop[:, None], states[torch.arange(bsz, device=device), node, :], final)
            node = torch.where(act == 1, node * 2, node)
            node = torch.where(act == 2, node * 2 + 1, node)
            node = node.clamp(1, self.nodes)
        route_loss = torch.stack(action_losses).mean() if action_losses else torch.tensor(0.0, device=device)
        return final, route_loss

    def soft_frontier_state(self, states: torch.Tensor, query_node: torch.Tensor) -> torch.Tensor:
        bsz = states.shape[0]
        device = states.device
        q = self.query_emb(query_node)
        masses = torch.zeros(bsz, self.nodes + 1, device=device)
        masses[:, 1] = 1.0
        result = torch.zeros(bsz, states.shape[-1], device=device)
        for _ in range(self.max_depth):
            new_masses = torch.zeros_like(masses)
            for node in range(1, self.nodes + 1):
                mass = masses[:, node]
                if float(mass.detach().abs().sum().cpu()) == 0.0:
                    continue
                node_ids = torch.full((bsz,), node, dtype=torch.long, device=device)
                probs = torch.softmax(self.action_logits_for_node(states, q, node_ids), dim=-1)
                h = states[:, node, :]
                result = result + (mass * probs[:, 0])[:, None] * h
                left = node * 2
                right = node * 2 + 1
                if left <= self.nodes:
                    new_masses[:, left] += mass * probs[:, 1]
                else:
                    result = result + (mass * probs[:, 1])[:, None] * h
                if right <= self.nodes:
                    new_masses[:, right] += mass * probs[:, 2]
                else:
                    result = result + (mass * probs[:, 2])[:, None] * h
            masses = new_masses
        # Force any non-stopped probability mass to contribute at its current node.
        for node in range(1, self.nodes + 1):
            mass = masses[:, node]
            if float(mass.detach().abs().sum().cpu()) != 0.0:
                result = result + mass[:, None] * states[:, node, :]
        return result

    @torch.no_grad()
    def hard_state_and_route(
        self,
        states: torch.Tensor,
        query_node: torch.Tensor,
        teacher_actions: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = states.shape[0]
        device = states.device
        q = self.query_emb(query_node)
        node = torch.ones(bsz, dtype=torch.long, device=device)
        final = states[:, 1, :]
        route_ok = torch.ones(bsz, dtype=torch.bool, device=device)
        done = torch.zeros(bsz, dtype=torch.bool, device=device)
        for t in range(self.max_depth):
            logits = self.action_logits_for_node(states, q, node)
            pred = logits.argmax(dim=-1)
            valid = teacher_mask[:, t] & (~done)
            route_ok = route_ok & (~valid | (pred == teacher_actions[:, t]))
            stop = pred == 0
            final = torch.where((stop & ~done)[:, None], states[torch.arange(bsz, device=device), node, :], final)
            done = done | stop
            node = torch.where((pred == 1) & (~done), node * 2, node)
            node = torch.where((pred == 2) & (~done), node * 2 + 1, node)
            node = node.clamp(1, self.nodes)
        return final, route_ok

    def forward(
        self,
        tokens: torch.Tensor,
        query_node: torch.Tensor,
        teacher_actions: torch.Tensor,
        teacher_mask: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.encoder(tokens)
        q = self.query_emb(query_node)
        if mode == "teacher":
            read_state, route_loss = self.teacher_forced_state(
                states, query_node, teacher_actions, teacher_mask
            )
        elif mode == "soft":
            read_state = self.soft_frontier_state(states, query_node)
            route_loss = torch.tensor(0.0, device=tokens.device)
        else:
            raise ValueError(mode)
        logits = self.readout(torch.cat([read_state, q], dim=-1))
        return logits, route_loss


def train_root(
    model: RootQueryDecoder,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> List[Dict[str, float]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trace: List[Dict[str, float]] = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            query = batch["query_node"].to(device)
            target = batch["target"].to(device)
            loss = F.cross_entropy(model(tokens, query), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * tokens.shape[0]
            n += tokens.shape[0]
        if ep in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": ep, "loss": total / max(1, n)})
    return trace


def train_read(
    model: ProbabilisticReadKernel,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    route_weight: float,
) -> List[Dict[str, float]]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    trace: List[Dict[str, float]] = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        route_total = 0.0
        n = 0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            query = batch["query_node"].to(device)
            target = batch["target"].to(device)
            actions = batch["teacher_actions"].to(device)
            mask = batch["teacher_mask"].to(device)
            logits, route_loss = model(tokens, query, actions, mask, mode="teacher")
            label_loss = F.cross_entropy(logits, target)
            loss = label_loss + route_weight * route_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * tokens.shape[0]
            route_total += float(route_loss.detach().cpu()) * tokens.shape[0]
            n += tokens.shape[0]
        if ep in {0, epochs // 2, epochs - 1}:
            trace.append(
                {
                    "epoch": ep,
                    "loss": total / max(1, n),
                    "route_loss": route_total / max(1, n),
                }
            )
    return trace


@torch.no_grad()
def eval_root(model: RootQueryDecoder, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    ok = 0
    internal_total = 0
    internal_ok = 0
    leaf_total = 0
    leaf_ok = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        query = batch["query_node"].to(device)
        target = batch["target"].to(device)
        is_internal = batch["is_internal"].to(device)
        pred = model(tokens, query).argmax(dim=-1)
        correct = pred == target
        total += int(target.numel())
        ok += int(correct.sum().item())
        internal_total += int(is_internal.sum().item())
        internal_ok += int((correct & is_internal).sum().item())
        leaf_total += int((~is_internal).sum().item())
        leaf_ok += int((correct & ~is_internal).sum().item())
    return {
        "acc": ok / max(1, total),
        "internal_acc": internal_ok / max(1, internal_total),
        "leaf_acc": leaf_ok / max(1, leaf_total),
    }


@torch.no_grad()
def eval_read(
    model: ProbabilisticReadKernel,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Dict[str, float]:
    model.eval()
    total = ok = 0
    internal_total = internal_ok = 0
    leaf_total = leaf_ok = 0
    route_total = route_ok_total = 0
    stop_internal_total = stop_internal_ok = 0
    stop_leaf_total = stop_leaf_ok = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        query = batch["query_node"].to(device)
        target = batch["target"].to(device)
        actions = batch["teacher_actions"].to(device)
        mask = batch["teacher_mask"].to(device)
        is_internal = batch["is_internal"].to(device)
        if mode == "hard":
            states = model.encoder(tokens)
            read_state, route_ok = model.hard_state_and_route(states, query, actions, mask)
            q = model.query_emb(query)
            logits = model.readout(torch.cat([read_state, q], dim=-1))
            route_total += int(route_ok.numel())
            route_ok_total += int(route_ok.sum().item())

            # Stop should occur at target depth. Approximate by full route equality.
            stop_internal_total += int(is_internal.sum().item())
            stop_internal_ok += int((route_ok & is_internal).sum().item())
            stop_leaf_total += int((~is_internal).sum().item())
            stop_leaf_ok += int((route_ok & ~is_internal).sum().item())
        else:
            logits, _ = model(tokens, query, actions, mask, mode="soft")
        pred = logits.argmax(dim=-1)
        correct = pred == target
        total += int(target.numel())
        ok += int(correct.sum().item())
        internal_total += int(is_internal.sum().item())
        internal_ok += int((correct & is_internal).sum().item())
        leaf_total += int((~is_internal).sum().item())
        leaf_ok += int((correct & ~is_internal).sum().item())

    out = {
        "acc": ok / max(1, total),
        "internal_acc": internal_ok / max(1, internal_total),
        "leaf_acc": leaf_ok / max(1, leaf_total),
    }
    if mode == "hard":
        out.update(
            {
                "route_acc": route_ok_total / max(1, route_total),
                "internal_stop_route_acc": stop_internal_ok / max(1, stop_internal_total),
                "leaf_stop_route_acc": stop_leaf_ok / max(1, stop_leaf_total),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmt", default="/mnt/nas/datasets/wmt17/train.zh-en")
    ap.add_argument("--spm", default="/mnt/nas/datasets/wmt17/sp_bpe.model")
    ap.add_argument("--out", default="ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--scan-lines", type=int, default=100000)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--vocab-limit", type=int, default=513)
    ap.add_argument("--checksum-buckets", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--route-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    start = time.time()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device.strip() if torch.cuda.is_available() else "cpu")

    seqs, examples = collect_sequences(
        Path(args.wmt),
        Path(args.spm),
        args.samples,
        args.min_len,
        args.max_len,
        args.vocab_limit,
        args.scan_lines,
    )
    random.Random(args.seed).shuffle(seqs)
    n_train = int(len(seqs) * 0.8)
    n_test = int(len(seqs) * 0.1)
    train_seqs = seqs[:n_train]
    test_seqs = seqs[n_train : n_train + n_test]
    ood_seqs = seqs[n_train + n_test :]

    max_depth = int(np.log2(args.max_len)) + 1
    labels = args.vocab_limit + args.checksum_buckets
    train_ds = NodeQueryDataset(
        train_seqs,
        args.max_len,
        args.vocab_limit,
        args.checksum_buckets,
        True,
        args.seed,
    )
    test_ds = NodeQueryDataset(
        test_seqs,
        args.max_len,
        args.vocab_limit,
        args.checksum_buckets,
        False,
        args.seed,
    )
    ood_ds = NodeQueryDataset(
        ood_seqs,
        args.max_len,
        args.vocab_limit,
        args.checksum_buckets,
        False,
        args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    tail_ok = all(tail_recursive_interpreter(n) == n for n in range(1, args.max_len * 2))

    root = RootQueryDecoder(args.vocab_limit, args.max_len, args.dim, labels)
    root_trace = train_root(root, train_loader, device, args.epochs, args.lr)
    root_test = eval_root(root, test_loader, device)
    root_ood = eval_root(root, ood_loader, device)

    read = ProbabilisticReadKernel(args.vocab_limit, args.max_len, args.dim, labels, max_depth)
    read_trace = train_read(
        read,
        train_loader,
        device,
        args.epochs,
        args.lr,
        args.route_weight,
    )
    read_test_hard = eval_read(read, test_loader, device, mode="hard")
    read_ood_hard = eval_read(read, ood_loader, device, mode="hard")
    read_test_soft = eval_read(read, test_loader, device, mode="soft")
    read_ood_soft = eval_read(read, ood_loader, device, mode="soft")

    improvement = read_ood_hard["acc"] - root_ood["acc"]
    pilot_pass = (
        tail_ok
        and read_ood_hard["acc"] >= 0.80
        and read_ood_hard["route_acc"] >= 0.95
        and improvement >= 0.20
        and read_ood_hard["internal_stop_route_acc"] >= 0.90
        and read_ood_hard["leaf_stop_route_acc"] >= 0.90
    )

    summary = {
        "claim": "S1-READ-C01",
        "predict": "P-S1-READ01",
        "host": "io.grepcode.cn",
        "device": str(device),
        "dataset": {
            "wmt_path": args.wmt,
            "spm_model": args.spm,
            "samples": len(seqs),
            "train": len(train_seqs),
            "test": len(test_seqs),
            "ood": len(ood_seqs),
            "min_len": args.min_len,
            "max_len": args.max_len,
            "vocab_limit_including_pad": args.vocab_limit,
            "checksum_buckets": args.checksum_buckets,
            "labels": labels,
            "examples": examples,
        },
        "algorithm": {
            "hard_read": "tail-recursive while loop from arr[1] over stop/left/right",
            "soft_read": "frontier dynamic program that accumulates stopped probability mass",
            "tail_recursive_interpreter_ok": tail_ok,
        },
        "models": {
            "root_query_decoder": {
                "parameters": sum(p.numel() for p in root.parameters()),
                "trace": root_trace,
                "test": root_test,
                "ood": root_ood,
            },
            "probabilistic_read_kernel": {
                "parameters": sum(p.numel() for p in read.parameters()),
                "trace": read_trace,
                "test_hard": read_test_hard,
                "ood_hard": read_ood_hard,
                "test_soft": read_test_soft,
                "ood_soft": read_ood_soft,
            },
        },
        "derived": {
            "ood_hard_acc_improvement_over_root": improvement,
            "read_vs_root_parameter_ratio": sum(p.numel() for p in read.parameters())
            / max(1, sum(p.numel() for p in root.parameters())),
        },
        "pilot_pass": pilot_pass,
        "interpretation": {
            "supported": "S1-READ-C01 is supported as a pilot if pilot_pass is true.",
            "not_proved": [
                "not translation",
                "not semantic world model",
                "not unsupervised routing",
                "not long-sequence syntax",
            ],
        },
        "elapsed_sec": round(time.time() - start, 3),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in root_trace:
            f.write(json.dumps({"model": "root_query_decoder", **row}) + "\n")
        for row in read_trace:
            f.write(json.dumps({"model": "probabilistic_read_kernel", **row}) + "\n")
    (out_dir / "README.md").write_text(
        "# S1 probabilistic read kernel probe\n\n"
        "Tests TreeHeap stop/left/right read collapse over real WMT short BPE sequences.\n\n"
        f"Decision: `S1-READ-C01 -> {'supported pilot' if pilot_pass else 'open/rejected pilot'}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
