"""Minimal TreeHeap/embedding fixed-point training probe.

One token table enters one shared TreeHeap repeatedly.  Corpus positive and
negative pairs score the resulting landings, and one loss updates both the
table and TreeHeap parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-9


@dataclass
class Config:
    data: str
    spm_model: str
    max_scan_lines: int
    vocab_size: int
    max_sentence_tokens: int
    window: int
    test_mod: int
    train_pair_cap: int
    test_pair_cap: int
    dim: int
    depth: int
    rounds: int
    mix: float
    steps: int
    batch_size: int
    negatives: int
    lr: float
    score_scale: float
    eval_pairs: int
    seed: int
    device: str


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_english(path: str, max_lines: int) -> Iterable[Tuple[int, bytes, str]]:
    with open(path, "rb") as handle:
        for line_index, raw in enumerate(handle):
            if line_index >= max_lines:
                break
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError:
                continue
            if len(fields) >= 2 and fields[1].strip():
                yield line_index, raw, fields[1]


def reservoir_add(
    reservoir: List[Tuple[int, int]],
    item: Tuple[int, int],
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        reservoir[replacement] = item


def prepare_pairs(cfg: Config, cache_path: Path) -> Dict[str, object]:
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    counts: Counter[int] = Counter()
    scan_digest = hashlib.sha256()
    valid_lines = 0
    token_total = 0
    for _line_index, raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        scan_digest.update(raw)
        ids = [i for i in sp.encode(text, out_type=int) if i >= 4][: cfg.max_sentence_tokens]
        counts.update(ids)
        valid_lines += 1
        token_total += len(ids)
    ranked = [token for token, _count in counts.most_common(cfg.vocab_size)]
    if len(ranked) < cfg.vocab_size:
        raise RuntimeError(f"only {len(ranked)} observed token types")
    token_map = {token: index for index, token in enumerate(ranked)}
    train_pairs: List[Tuple[int, int]] = []
    test_pairs: List[Tuple[int, int]] = []
    train_seen = 0
    test_seen = 0
    train_rng = random.Random(cfg.seed + 101)
    test_rng = random.Random(cfg.seed + 102)
    for line_index, _raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        ids = [token_map[i] for i in sp.encode(text, out_type=int) if i in token_map]
        ids = ids[: cfg.max_sentence_tokens]
        is_test = line_index % cfg.test_mod == 0
        for pos, center in enumerate(ids):
            for other in range(max(0, pos - cfg.window), min(len(ids), pos + cfg.window + 1)):
                if other == pos:
                    continue
                pair = (center, ids[other])
                if is_test:
                    test_seen += 1
                    reservoir_add(test_pairs, pair, test_seen, cfg.test_pair_cap, test_rng)
                else:
                    train_seen += 1
                    reservoir_add(train_pairs, pair, train_seen, cfg.train_pair_cap, train_rng)
    frequency = torch.tensor([counts[token] for token in ranked], dtype=torch.float32)
    payload = {
        "train_pairs": torch.tensor(train_pairs, dtype=torch.long),
        "test_pairs": torch.tensor(test_pairs, dtype=torch.long),
        "frequency": frequency,
        "token_ids": ranked,
        "pieces": [sp.id_to_piece(token) for token in ranked],
        "meta": {
            "max_scan_lines": cfg.max_scan_lines,
            "vocab_size": cfg.vocab_size,
            "max_sentence_tokens": cfg.max_sentence_tokens,
            "window": cfg.window,
            "test_mod": cfg.test_mod,
            "valid_lines": valid_lines,
            "token_total": token_total,
            "train_pairs_seen": train_seen,
            "test_pairs_seen": test_seen,
            "train_pairs_kept": len(train_pairs),
            "test_pairs_kept": len(test_pairs),
            "scan_sha256": scan_digest.hexdigest(),
            "spm_sha256": sha256(cfg.spm_model),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def load_or_prepare(cfg: Config, cache_path: Path) -> Dict[str, object]:
    if not cache_path.exists():
        return prepare_pairs(cfg, cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    expected = {
        "max_scan_lines": cfg.max_scan_lines,
        "vocab_size": cfg.vocab_size,
        "max_sentence_tokens": cfg.max_sentence_tokens,
        "window": cfg.window,
        "test_mod": cfg.test_mod,
    }
    actual = {key: payload["meta"][key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"pair cache contract mismatch: {actual} != {expected}")
    return payload


class EmbeddingOnly(nn.Module):
    def __init__(self, initial: torch.Tensor):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(initial.clone(), freeze=False)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(ids), dim=-1)


class RecurrentTreeHeap(nn.Module):
    def __init__(self, initial: torch.Tensor, depth: int, rounds: int, mix: float):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(initial.clone(), freeze=False)
        self.depth = depth
        self.rounds = rounds
        self.mix = mix
        dim = initial.shape[1]
        internal = 2**depth - 1
        nonroot = 2 ** (depth + 1) - 2
        self.route_weight = nn.Parameter(torch.empty(internal, dim))
        self.route_bias = nn.Parameter(torch.zeros(internal))
        self.node_value = nn.Parameter(torch.empty(nonroot, dim))
        nn.init.normal_(self.route_weight, std=0.02)
        nn.init.normal_(self.node_value, std=0.02)

    def fall_once(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, dim = state.shape
        mass = torch.ones((batch, 1), device=state.device, dtype=state.dtype)
        read = torch.zeros_like(state)
        internal_offset = 0
        value_offset = 0
        for level in range(self.depth):
            nodes = 2**level
            weight = self.route_weight[internal_offset : internal_offset + nodes]
            bias = self.route_bias[internal_offset : internal_offset + nodes]
            logits = (state @ weight.t()) / math.sqrt(dim) + bias
            right = logits.sigmoid()
            next_mass = torch.stack((mass * (1.0 - right), mass * right), dim=-1).reshape(batch, -1)
            child_values = self.node_value[value_offset : value_offset + 2 * nodes]
            read = read + next_mass @ child_values
            mass = next_mass
            internal_offset += nodes
            value_offset += 2 * nodes
        updated = F.normalize((1.0 - self.mix) * state + self.mix * read / self.depth, dim=-1)
        return updated, mass

    def encode(
        self, ids: torch.Tensor, return_round_leaves: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        state = F.normalize(self.embedding(ids), dim=-1)
        round_leaves: List[torch.Tensor] = []
        leaf_mass = torch.empty(0, device=ids.device)
        for _ in range(self.rounds):
            state, leaf_mass = self.fall_once(state)
            if return_round_leaves:
                round_leaves.append(leaf_mass.argmax(dim=1))
        return state, leaf_mass, round_leaves


def pair_loss(
    center: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    positive_score = (center * positive).sum(dim=-1) * scale
    negative_score = (center[:, None, :] * negative).sum(dim=-1) * scale
    return -(F.logsigmoid(positive_score).mean() + F.logsigmoid(-negative_score).mean())


def encode_batch(
    model: nn.Module,
    center: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_ids = torch.cat((center, positive, negative.reshape(-1)))
    unique, inverse = torch.unique(all_ids, sorted=True, return_inverse=True)
    if isinstance(model, RecurrentTreeHeap):
        encoded, _mass, _leaves = model.encode(unique)
    else:
        encoded = model.encode(unique)
    restored = encoded[inverse]
    batch = len(center)
    return (
        restored[:batch],
        restored[batch : 2 * batch],
        restored[2 * batch :].view(batch, negative.shape[1], -1),
    )


def lcp_depth(a: torch.Tensor, b: torch.Tensor, depth: int) -> torch.Tensor:
    same = []
    for shift in range(depth - 1, -1, -1):
        same.append(((a >> shift) & 1) == ((b >> shift) & 1))
    bits = torch.stack(same, dim=1)
    return bits.to(torch.long).cumprod(dim=1).sum(dim=1)


@torch.no_grad()
def materialize_tree(
    model: RecurrentTreeHeap, vocab_size: int, device: torch.device, chunk: int = 512
) -> Dict[str, torch.Tensor]:
    states, leaves, per_round = [], [], [[] for _ in range(model.rounds)]
    for start in range(0, vocab_size, chunk):
        ids = torch.arange(start, min(vocab_size, start + chunk), device=device)
        state, mass, round_leaves = model.encode(ids, return_round_leaves=True)
        states.append(state.cpu())
        leaves.append(mass.argmax(dim=1).cpu())
        for index, values in enumerate(round_leaves):
            per_round[index].append(values.cpu())
    return {
        "embedding": torch.cat(states),
        "leaf": torch.cat(leaves),
        "round_leaf": torch.stack([torch.cat(parts) for parts in per_round]),
    }


@torch.no_grad()
def evaluate(
    baseline: EmbeddingOnly,
    tree: RecurrentTreeHeap,
    test_pairs: torch.Tensor,
    frequency: torch.Tensor,
    cfg: Config,
) -> Dict[str, object]:
    device = torch.device(cfg.device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 9001)
    count = min(cfg.eval_pairs, len(test_pairs))
    chosen = torch.randperm(len(test_pairs), generator=generator)[:count]
    pairs = test_pairs[chosen]
    noise = frequency.pow(0.75)
    noise /= noise.sum()
    negative = torch.multinomial(noise, count, replacement=True, generator=generator)
    center, positive, negative = pairs[:, 0].to(device), pairs[:, 1].to(device), negative.to(device)
    b_center = baseline.encode(center)
    b_positive = baseline.encode(positive)
    b_negative = baseline.encode(negative)
    t_center, _cmass, _ = tree.encode(center)
    t_positive, _pmass, _ = tree.encode(positive)
    t_negative, _nmass, _ = tree.encode(negative)
    baseline_pos = (b_center * b_positive).sum(dim=1)
    baseline_neg = (b_center * b_negative).sum(dim=1)
    tree_pos = (t_center * t_positive).sum(dim=1)
    tree_neg = (t_center * t_negative).sum(dim=1)
    materialized = materialize_tree(tree, cfg.vocab_size, device)
    leaf = materialized["leaf"]
    counts = torch.bincount(leaf, minlength=2**cfg.depth).to(torch.float64)
    occupied = counts[counts > 0]
    probability = occupied / occupied.sum()
    entropy = float((-(probability * probability.log()).sum() / math.log(2**cfg.depth)).item())
    center_leaf = leaf[center.cpu()]
    positive_leaf = leaf[positive.cpu()]
    negative_leaf = leaf[negative.cpu()]
    positive_lcp = lcp_depth(center_leaf, positive_leaf, cfg.depth).to(torch.float32)
    negative_lcp = lcp_depth(center_leaf, negative_leaf, cfg.depth).to(torch.float32)
    round_leaf = materialized["round_leaf"]
    round_flip = []
    for index in range(1, len(round_leaf)):
        round_flip.append(float((round_leaf[index] != round_leaf[index - 1]).to(torch.float32).mean().item()))
    return {
        "pairs": count,
        "baseline_pair_accuracy": float((baseline_pos > baseline_neg).to(torch.float32).mean().item()),
        "tree_pair_accuracy": float((tree_pos > tree_neg).to(torch.float32).mean().item()),
        "baseline_margin": float((baseline_pos - baseline_neg).mean().item()),
        "tree_margin": float((tree_pos - tree_neg).mean().item()),
        "leaf_utilization": float(len(occupied) / (2**cfg.depth)),
        "occupancy_entropy": entropy,
        "positive_lcp": float(positive_lcp.mean().item()),
        "negative_lcp": float(negative_lcp.mean().item()),
        "lcp_margin": float((positive_lcp - negative_lcp).mean().item()),
        "round_flip_rate": round_flip,
        "embedding_table": materialized["embedding"],
        "leaf_table": leaf,
    }


def train(cfg: Config, payload: Dict[str, object], evidence: Path) -> Dict[str, object]:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    initial = torch.empty((cfg.vocab_size, cfg.dim), device=device)
    nn.init.normal_(initial, std=0.02)
    baseline = EmbeddingOnly(initial).to(device)
    tree = RecurrentTreeHeap(initial, cfg.depth, cfg.rounds, cfg.mix).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=cfg.lr)
    tree_optimizer = torch.optim.AdamW(tree.parameters(), lr=cfg.lr)
    train_pairs = payload["train_pairs"]
    frequency = payload["frequency"]
    noise = frequency.pow(0.75)
    noise /= noise.sum()
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 3001)
    trace = []
    first_gradients = None
    started = time.time()
    for step in range(1, cfg.steps + 1):
        pair_index = torch.randint(0, len(train_pairs), (cfg.batch_size,), generator=generator)
        batch = train_pairs[pair_index]
        negative = torch.multinomial(
            noise, cfg.batch_size * cfg.negatives, replacement=True, generator=generator
        ).view(cfg.batch_size, cfg.negatives)
        center, positive = batch[:, 0].to(device), batch[:, 1].to(device)
        negative = negative.to(device)

        bc, bp, bn = encode_batch(baseline, center, positive, negative)
        baseline_loss = pair_loss(bc, bp, bn, cfg.score_scale)
        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss.backward()
        baseline_optimizer.step()

        tc, tp, tn = encode_batch(tree, center, positive, negative)
        tree_loss = pair_loss(tc, tp, tn, cfg.score_scale)
        tree_optimizer.zero_grad(set_to_none=True)
        tree_loss.backward()
        if first_gradients is None:
            first_gradients = {
                "embedding": float(tree.embedding.weight.grad.norm().item()),
                "route_weight": float(tree.route_weight.grad.norm().item()),
                "route_bias": float(tree.route_bias.grad.norm().item()),
                "node_value": float(tree.node_value.grad.norm().item()),
            }
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 5.0)
        tree_optimizer.step()
        if step == 1 or step == cfg.steps or step % max(1, cfg.steps // 10) == 0:
            row = {
                "step": step,
                "baseline_loss": float(baseline_loss.detach().item()),
                "tree_loss": float(tree_loss.detach().item()),
                "elapsed_seconds": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row), flush=True)

    result = evaluate(baseline, tree, payload["test_pairs"], frequency, cfg)
    embedding_table = result.pop("embedding_table")
    leaf_table = result.pop("leaf_table")
    torch.save(
        {
            "embedding": embedding_table,
            "leaf": leaf_table,
            "token_ids": payload["token_ids"],
            "pieces": payload["pieces"],
            "config": asdict(cfg),
        },
        evidence / "embedding_lookup.pt",
    )
    torch.save(tree.state_dict(), evidence / "treeheap_state.pt")
    return {
        "trace": trace,
        "first_gradients": first_gradients,
        "evaluation": result,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    ap.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    ap.add_argument("--pair-cache", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--max-scan-lines", type=int, default=100_000)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--max-sentence-tokens", type=int, default=96)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--test-mod", type=int, default=10)
    ap.add_argument("--train-pair-cap", type=int, default=1_000_000)
    ap.add_argument("--test-pair-cap", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--negatives", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--score-scale", type=float, default=5.0)
    ap.add_argument("--eval-pairs", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=19301)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = Config(
        data=args.data, spm_model=args.spm_model, max_scan_lines=args.max_scan_lines,
        vocab_size=args.vocab_size, max_sentence_tokens=args.max_sentence_tokens,
        window=args.window, test_mod=args.test_mod, train_pair_cap=args.train_pair_cap,
        test_pair_cap=args.test_pair_cap, dim=args.dim, depth=args.depth,
        rounds=args.rounds, mix=args.mix, steps=args.steps, batch_size=args.batch_size,
        negatives=args.negatives, lr=args.lr, score_scale=args.score_scale,
        eval_pairs=args.eval_pairs, seed=args.seed, device=args.device,
    )
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.pair_cache)
    payload = load_or_prepare(cfg, cache_path)
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    if args.prepare_only:
        summary = {"claim": "S1-TH-EMBED-FP-C01", "config": asdict(cfg), "cache": payload["meta"]}
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return
    trained = train(cfg, payload, evidence)
    evaluation = trained["evaluation"]
    gradients = trained["first_gradients"]
    gates = {
        "all_gradients_finite_positive": all(math.isfinite(v) and v > 1e-8 for v in gradients.values()),
        "tree_pair_accuracy_gt_0_52": evaluation["tree_pair_accuracy"] > 0.52,
        "positive_lcp_gt_negative_lcp": evaluation["lcp_margin"] > 0.0,
        "leaf_utilization_gt_0_10": evaluation["leaf_utilization"] > 0.10,
        "occupancy_entropy_gt_0_50": evaluation["occupancy_entropy"] > 0.50,
        "finite_losses": all(
            math.isfinite(row[key])
            for row in trained["trace"]
            for key in ("baseline_loss", "tree_loss")
        ),
    }
    summary = {
        "claim": "S1-TH-EMBED-FP-C01",
        "experiment": "P-S1-TH-EMBED-FP01",
        "config": asdict(cfg),
        "cache": payload["meta"],
        **trained,
        "gates": gates,
        "smoke_pass": all(gates.values()),
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trained["trace"]), encoding="utf-8"
    )
    (evidence / "README.md").write_text(
        "# TH Embedding Fixed Point\n\n```json\n"
        + json.dumps({"evaluation": evaluation, "gradients": gradients, "gates": gates}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"evaluation": evaluation, "gradients": gradients, "gates": gates, "smoke_pass": summary["smoke_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
