"""Build large TreeHeap token landmarks and audit background association."""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F


EPS = 1e-9


@dataclass
class Config:
    data: str
    spm_model: str
    simlex: str
    max_scan_lines: int
    target_vocab: int
    context_vocab: int
    window: int
    max_sentence_tokens: int
    test_mod: int
    depth: int
    bootstrap_seeds: int
    seed_start: int
    temperatures: List[float]
    em_steps: int
    power_steps: int
    alpha: float
    sgns_dim: int
    sgns_steps: int
    sgns_batch: int
    sgns_negatives: int
    sgns_lr: float
    device: str


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_english(path: str, max_lines: int) -> Iterable[Tuple[int, bytes, str]]:
    with open(path, "rb") as handle:
        for index, raw in enumerate(handle):
            if index >= max_lines:
                break
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError:
                continue
            if len(fields) >= 2 and fields[1].strip():
                yield index, raw, fields[1]


def choose_vocab(cfg: Config, sp: spm.SentencePieceProcessor) -> Dict[str, object]:
    counts: Counter[int] = Counter()
    digest = hashlib.sha256()
    valid_lines = 0
    token_total = 0
    for _index, raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        digest.update(raw)
        ids = [i for i in sp.encode(text, out_type=int) if i >= 4][: cfg.max_sentence_tokens]
        counts.update(ids)
        valid_lines += 1
        token_total += len(ids)
    ranked = [token for token, _ in counts.most_common()]
    if len(ranked) < max(cfg.target_vocab, cfg.context_vocab):
        raise RuntimeError("insufficient observed vocabulary")
    return {
        "counts": counts,
        "target_ids": ranked[: cfg.target_vocab],
        "context_ids": ranked[: cfg.context_vocab],
        "valid_lines": valid_lines,
        "token_total": token_total,
        "scan_sha256": digest.hexdigest(),
    }


def build_counts(
    cfg: Config,
    sp: spm.SentencePieceProcessor,
    target_ids: Sequence[int],
    context_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    target_map = {token: i for i, token in enumerate(target_ids)}
    context_map = {token: i for i, token in enumerate(context_ids)}
    train = torch.zeros((len(target_ids), len(context_ids)), dtype=torch.float32)
    test = torch.zeros_like(train)
    stats = {"train_lines": 0, "test_lines": 0, "train_pairs": 0, "test_pairs": 0}
    for line_index, _raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        ids = [i for i in sp.encode(text, out_type=int) if i >= 4][: cfg.max_sentence_tokens]
        is_test = line_index % cfg.test_mod == 0
        matrix = test if is_test else train
        pairs = 0
        for pos, token in enumerate(ids):
            center = target_map.get(token)
            if center is None:
                continue
            for other in range(max(0, pos - cfg.window), min(len(ids), pos + cfg.window + 1)):
                if other == pos:
                    continue
                context = context_map.get(ids[other])
                if context is not None:
                    matrix[center, context] += 1.0
                    pairs += 1
        key = "test" if is_test else "train"
        stats[f"{key}_lines"] += 1
        stats[f"{key}_pairs"] += pairs
    return train, test, stats


def distributions(counts: torch.Tensor, alpha: float) -> torch.Tensor:
    smooth = counts + alpha
    return smooth / smooth.sum(dim=1, keepdim=True).clamp_min(EPS)


def deterministic_axis_power(
    x: torch.Tensor, weights: torch.Tensor, steps: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = (weights[:, None] * x).sum(dim=0) / weights.sum().clamp_min(EPS)
    centered = (x - center) * weights.sqrt()[:, None]
    row = torch.arange(len(x), device=x.device, dtype=torch.int64)
    signs = (((row * 1103515245 + 12345) >> 8) & 1).to(x.dtype) * 2.0 - 1.0
    axis = centered.t() @ signs
    if float(axis.norm().item()) < EPS:
        axis = centered[0].clone()
    axis /= axis.norm().clamp_min(EPS)
    for _ in range(steps):
        axis = centered.t() @ (centered @ axis)
        axis /= axis.norm().clamp_min(EPS)
    pivot = int(axis.abs().argmax().item())
    if axis[pivot] < 0:
        axis = -axis
    return center, axis


def split_node(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
    local = x[indices]
    w = weights[indices]
    center, axis = deterministic_axis_power(local, w, cfg.power_steps)
    projection = (local - center) @ axis
    order = projection.argsort()
    half = max(1, len(indices) // 2)
    hard = torch.zeros(len(indices), dtype=torch.long, device=x.device)
    hard[order[half:]] = 1
    centers = torch.stack(
        [
            (w[hard == side, None] * local[hard == side]).sum(dim=0)
            / w[hard == side].sum().clamp_min(EPS)
            for side in (0, 1)
        ]
    )
    resp = F.one_hot(hard, num_classes=2).to(local.dtype)
    base = ((local - center) ** 2).sum(dim=1).median().clamp_min(1e-5)
    for ratio in cfg.temperatures:
        temperature = base * ratio
        for _ in range(cfg.em_steps):
            distance = ((local[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)
            mass = (w[:, None] * resp).sum(dim=0).clamp_min(EPS)
            logits = -distance / temperature + (mass / mass.sum()).log()[None, :]
            resp = logits.softmax(dim=1)
            wr = w[:, None] * resp
            centers = (wr.t() @ local) / wr.sum(dim=0)[:, None].clamp_min(EPS)
    hard = resp.argmax(dim=1)
    used_fallback = False
    if not (hard == 0).any() or not (hard == 1).any():
        used_fallback = True
        hard.zero_()
        hard[order[half:]] = 1
        # A forced structural split must also be the published route.  Keeping
        # the collapsed soft responsibility here creates two incompatible
        # landing coordinates: one for leaf NLL and another for route distance.
        resp = F.one_hot(hard, num_classes=2).to(local.dtype)
    child_mass = torch.stack([w[hard == side].sum() for side in (0, 1)])
    child_center = torch.stack(
        [
            (w[hard == side, None] * local[hard == side]).sum(dim=0)
            / w[hard == side].sum().clamp_min(EPS)
            for side in (0, 1)
        ]
    )
    folded = (child_mass[:, None] * child_center).sum(dim=0) / child_mass.sum()
    return hard, resp, float((folded - center).abs().max().item()), used_fallback


def build_tree(x: torch.Tensor, weights: torch.Tensor, cfg: Config) -> Dict[str, object]:
    n = len(x)
    paths = torch.zeros((n, cfg.depth), dtype=torch.long, device=x.device)
    route = torch.full((n, cfg.depth), 0.5, dtype=x.dtype, device=x.device)
    groups = [torch.arange(n, device=x.device)]
    errors: List[float] = []
    fallback_splits = 0
    for level in range(cfg.depth):
        next_groups = []
        for indices in groups:
            if len(indices) <= 1:
                next_groups.extend([indices, indices.new_empty(0)])
                continue
            hard, resp, error, used_fallback = split_node(x, indices, weights, cfg)
            paths[indices, level] = hard
            route[indices, level] = resp[:, 1]
            errors.append(error)
            fallback_splits += int(used_fallback)
            next_groups.extend([indices[hard == 0], indices[hard == 1]])
        groups = next_groups
    leaf = torch.zeros(n, dtype=torch.long, device=x.device)
    for level in range(cfg.depth):
        leaf = leaf * 2 + paths[:, level]
    return {
        "leaf": leaf,
        "route": route,
        "max_conservation": max(errors, default=0.0),
        "fallback_splits": fallback_splits,
    }


def random_tree(n: int, depth: int, seed: int, device: torch.device) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    paths = torch.randint(0, 2, (n, depth), generator=generator).to(device)
    leaf = torch.zeros(n, dtype=torch.long, device=device)
    for level in range(depth):
        leaf = leaf * 2 + paths[:, level]
    return {"leaf": leaf, "route": paths.to(torch.float32)}


def leaf_nll(
    train: torch.Tensor, test: torch.Tensor, leaf: torch.Tensor, leaves: int, alpha: float
) -> float:
    prototype = torch.zeros((leaves, train.shape[1]), device=train.device, dtype=train.dtype)
    prototype.index_add_(0, leaf, train)
    prototype += alpha
    prototype /= prototype.sum(dim=1, keepdim=True).clamp_min(EPS)
    return float((-(test * prototype[leaf].clamp_min(EPS).log()).sum() / test.sum()).item())


def global_nll(train: torch.Tensor, test: torch.Tensor, alpha: float) -> float:
    probability = train.sum(dim=0) + alpha
    probability /= probability.sum()
    return float((-(test * probability.clamp_min(EPS).log()[None, :]).sum() / test.sum()).item())


def occupancy(leaf: torch.Tensor, leaves: int) -> Tuple[float, float]:
    counts = torch.bincount(leaf, minlength=leaves).to(torch.float64)
    occupied = counts[counts > 0]
    p = occupied / occupied.sum()
    return len(occupied) / leaves, float((-(p * p.log()).sum() / math.log(leaves)).item())


def sampled_partition_f1(a: torch.Tensor, b: torch.Tensor, seed: int, samples: int = 250_000) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(a)
    i = torch.randint(0, n, (samples,), generator=generator)
    j = torch.randint(0, n, (samples,), generator=generator)
    valid = i != j
    i, j = i[valid], j[valid]
    pa = a.cpu()[i] == a.cpu()[j]
    pb = b.cpu()[i] == b.cpu()[j]
    tp = int((pa & pb).sum().item())
    fp = int((pa & ~pb).sum().item())
    fn = int((~pa & pb).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(EPS, precision + recall)


def train_sgns(counts: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, Dict[str, object]]:
    device = torch.device(cfg.device)
    data = counts.to(device)
    row_mass = data.sum(dim=1).clamp_min(1.0)
    center_prob = row_mass / row_mass.sum()
    context_mass = data.sum(dim=0).clamp_min(1.0)
    noise = context_mass.pow(0.75)
    noise /= noise.sum()
    torch.manual_seed(cfg.seed_start + 500_000)
    center = torch.nn.Embedding(data.shape[0], cfg.sgns_dim, device=device)
    context = torch.nn.Embedding(data.shape[1], cfg.sgns_dim, device=device)
    torch.nn.init.normal_(center.weight, std=0.02)
    torch.nn.init.normal_(context.weight, std=0.02)
    optimizer = torch.optim.AdamW(list(center.parameters()) + list(context.parameters()), lr=cfg.sgns_lr)
    generator = torch.Generator(device=device).manual_seed(cfg.seed_start + 500_001)
    trace = []
    for step in range(cfg.sgns_steps):
        centers = torch.multinomial(center_prob, cfg.sgns_batch, replacement=True, generator=generator)
        positive = torch.multinomial(data[centers], 1, replacement=True, generator=generator).squeeze(1)
        negative = torch.multinomial(
            noise, cfg.sgns_batch * cfg.sgns_negatives, replacement=True, generator=generator
        ).view(cfg.sgns_batch, cfg.sgns_negatives)
        u = center(centers)
        positive_score = (u * context(positive)).sum(dim=1)
        negative_score = (u[:, None, :] * context(negative)).sum(dim=2)
        loss = -(F.logsigmoid(positive_score).mean() + F.logsigmoid(-negative_score).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, cfg.sgns_steps - 1) or (step + 1) % 250 == 0:
            trace.append({"step": step + 1, "loss": float(loss.detach().item())})
    return F.normalize(center.weight.detach(), dim=1), {"trace": trace}


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + end - 1) / 2.0 + 1.0
        for index in order[pos:end]:
            ranks[index] = rank
        pos = end
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / max(EPS, denominator)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(average_ranks(a), average_ranks(b))


def load_simlex(
    path: str, sp: spm.SentencePieceProcessor, target_map: Dict[int, int]
) -> List[Dict[str, object]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ids1 = [i for i in sp.encode(row["word1"], out_type=int) if i >= 4]
            ids2 = [i for i in sp.encode(row["word2"], out_type=int) if i >= 4]
            if len(ids1) == 1 and len(ids2) == 1 and ids1[0] in target_map and ids2[0] in target_map:
                rows.append(
                    {
                        "word1": row["word1"], "word2": row["word2"], "pos": row["POS"],
                        "human": float(row["SimLex999"]), "i": target_map[ids1[0]], "j": target_map[ids2[0]],
                    }
                )
    return rows


def benchmark_simlex(
    rows: Sequence[Dict[str, object]],
    route: torch.Tensor,
    random_route: torch.Tensor,
    x: torch.Tensor,
    sgns: torch.Tensor,
    frequency: torch.Tensor,
) -> Dict[str, object]:
    metrics = {"tree": [], "random": [], "hellinger": [], "sgns": [], "frequency": []}
    human = []
    positions = []
    for row in rows:
        i, j = int(row["i"]), int(row["j"])
        human.append(float(row["human"]))
        positions.append(str(row["pos"]))
        metrics["tree"].append(-float(torch.linalg.vector_norm(route[i] - route[j]).item()))
        metrics["random"].append(-float(torch.linalg.vector_norm(random_route[i] - random_route[j]).item()))
        metrics["hellinger"].append(-float(torch.linalg.vector_norm(x[i] - x[j]).item()))
        metrics["sgns"].append(float((sgns[i] * sgns[j]).sum().item()))
        metrics["frequency"].append(-abs(float(frequency[i].log().item() - frequency[j].log().item())))
    result = {"pairs": len(rows), "rho": {name: spearman(human, score) for name, score in metrics.items()}}
    result["by_pos"] = {}
    for pos in ("N", "V", "A"):
        selected = [index for index, value in enumerate(positions) if value == pos]
        result["by_pos"][pos] = {
            "pairs": len(selected),
            "rho": {
                name: spearman([human[i] for i in selected], [score[i] for i in selected])
                for name, score in metrics.items()
            },
        }
    return result


def nearest_examples(
    pieces: Sequence[str], route: torch.Tensor, sgns: torch.Tensor, frequency: torch.Tensor
) -> List[Dict[str, object]]:
    anchors = ["▁bank", "▁water", "▁people", "▁time", "▁doctor", "▁school", "▁government", "▁computer"]
    piece_map = {piece: index for index, piece in enumerate(pieces)}
    rows = []
    for piece in anchors:
        if piece not in piece_map:
            continue
        index = piece_map[piece]
        tree_distance = torch.linalg.vector_norm(route - route[index], dim=1)
        tree_distance[index] = float("inf")
        sgns_score = sgns @ sgns[index]
        sgns_score[index] = -float("inf")
        rows.append(
            {
                "token": piece,
                "frequency": int(frequency[index].item()),
                "tree": [pieces[i] for i in tree_distance.topk(10, largest=False).indices.cpu().tolist()],
                "sgns": [pieces[i] for i in sgns_score.topk(10).indices.cpu().tolist()],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    ap.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    ap.add_argument("--simlex", default="ara/s1-echo/data/simlex999/SimLex-999.txt")
    ap.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_treeheap_background_landmarks")
    ap.add_argument("--max-scan-lines", type=int, default=200_000)
    ap.add_argument("--target-vocab", type=int, default=8192)
    ap.add_argument("--context-vocab", type=int, default=4096)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--max-sentence-tokens", type=int, default=96)
    ap.add_argument("--test-mod", type=int, default=10)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--bootstrap-seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=19201)
    ap.add_argument("--temperatures", default="2,1,0.5,0.25,0.125")
    ap.add_argument("--em-steps", type=int, default=8)
    ap.add_argument("--power-steps", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sgns-dim", type=int, default=64)
    ap.add_argument("--sgns-steps", type=int, default=1500)
    ap.add_argument("--sgns-batch", type=int, default=1024)
    ap.add_argument("--sgns-negatives", type=int, default=8)
    ap.add_argument("--sgns-lr", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = Config(
        data=args.data, spm_model=args.spm_model, simlex=args.simlex,
        max_scan_lines=args.max_scan_lines, target_vocab=args.target_vocab,
        context_vocab=args.context_vocab, window=args.window,
        max_sentence_tokens=args.max_sentence_tokens, test_mod=args.test_mod,
        depth=args.depth, bootstrap_seeds=args.bootstrap_seeds, seed_start=args.seed_start,
        temperatures=[float(v) for v in args.temperatures.split(",")], em_steps=args.em_steps,
        power_steps=args.power_steps, alpha=args.alpha, sgns_dim=args.sgns_dim,
        sgns_steps=args.sgns_steps, sgns_batch=args.sgns_batch,
        sgns_negatives=args.sgns_negatives, sgns_lr=args.sgns_lr, device=args.device,
    )
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    vocab = choose_vocab(cfg, sp)
    counts = vocab.pop("counts")
    target_ids = vocab.pop("target_ids")
    context_ids = vocab.pop("context_ids")
    target_map = {token: index for index, token in enumerate(target_ids)}
    pieces = [sp.id_to_piece(token) for token in target_ids]
    train_cpu, test_cpu, count_stats = build_counts(cfg, sp, target_ids, context_ids)
    device = torch.device(cfg.device)
    test = test_cpu.to(device)
    simlex_rows = load_simlex(cfg.simlex, sp, target_map)
    sgns, sgns_meta = train_sgns(train_cpu, cfg)
    builds = []
    routes = []
    leaves = []
    random_leaves = []
    first_x = None
    first_random_route = None
    for offset in range(cfg.bootstrap_seeds):
        seed = cfg.seed_start + offset
        generator = torch.Generator(device="cpu").manual_seed(seed)
        observed = torch.poisson(train_cpu, generator=generator).to(device)
        frequency = observed.sum(dim=1).clamp_min(1.0)
        x = distributions(observed, cfg.alpha).sqrt()
        tree = build_tree(x, frequency, cfg)
        random_state = random_tree(cfg.target_vocab, cfg.depth, seed + 100_000, device)
        util, entropy = occupancy(tree["leaf"], 2**cfg.depth)
        random_util, random_entropy = occupancy(random_state["leaf"], 2**cfg.depth)
        row = {
            "seed": seed,
            "tree_nll": leaf_nll(observed, test, tree["leaf"], 2**cfg.depth, cfg.alpha),
            "random_nll": leaf_nll(observed, test, random_state["leaf"], 2**cfg.depth, cfg.alpha),
            "global_nll": global_nll(observed, test, cfg.alpha),
            "utilization": util,
            "occupancy_entropy": entropy,
            "random_utilization": random_util,
            "random_occupancy_entropy": random_entropy,
            "max_conservation": tree["max_conservation"],
            "fallback_splits": tree["fallback_splits"],
        }
        builds.append(row)
        routes.append(tree["route"].detach().cpu())
        leaves.append(tree["leaf"].detach().cpu())
        random_leaves.append(random_state["leaf"].detach().cpu())
        if offset == 0:
            first_x = x.detach()
            first_random_route = random_state["route"].detach()
        print(
            f"seed={seed} tree_nll={row['tree_nll']:.6f} random_nll={row['random_nll']:.6f} "
            f"util={util:.4f} entropy={entropy:.4f}", flush=True,
        )
    tree_stability = statistics.fmean(
        sampled_partition_f1(leaves[i], leaves[j], cfg.seed_start + i * 100 + j)
        for i in range(len(leaves)) for j in range(i + 1, len(leaves))
    )
    random_stability = statistics.fmean(
        sampled_partition_f1(random_leaves[i], random_leaves[j], cfg.seed_start + 1000 + i * 100 + j)
        for i in range(len(random_leaves)) for j in range(i + 1, len(random_leaves))
    )
    frequency = train_cpu.sum(dim=1).clamp_min(1.0).to(device)
    simlex = benchmark_simlex(
        simlex_rows, routes[0].to(device), first_random_route, first_x, sgns, frequency
    )
    nll = {
        "tree": mean_std([row["tree_nll"] for row in builds]),
        "random": mean_std([row["random_nll"] for row in builds]),
        "global": mean_std([row["global_nll"] for row in builds]),
    }
    utilization = mean_std([row["utilization"] for row in builds])
    entropy = mean_std([row["occupancy_entropy"] for row in builds])
    conservation = max(row["max_conservation"] for row in builds)
    rho = simlex["rho"]
    retention = rho["tree"] / rho["sgns"] if rho["sgns"] > 0 else float("nan")
    gates = {
        "simlex_covered_pairs_ge_300": simlex["pairs"] >= 300,
        "tree_simlex_rho_ge_0_12": rho["tree"] >= 0.12,
        "tree_minus_random_rho_ge_0_05": rho["tree"] - rho["random"] >= 0.05,
        "tree_minus_frequency_rho_ge_0_03": rho["tree"] - rho["frequency"] >= 0.03,
        "tree_sgns_retention_ge_0_35": retention >= 0.35,
        "tree_nll_better_than_global": nll["tree"]["mean"] < nll["global"]["mean"],
        "tree_nll_better_than_random": nll["tree"]["mean"] < nll["random"]["mean"],
        "tree_stability_better_than_random": tree_stability > random_stability,
        "mean_utilization_ge_0_75": utilization["mean"] >= 0.75,
        "mean_occupancy_entropy_ge_0_75": entropy["mean"] >= 0.75,
        "conservation_le_1e_5": conservation <= 1e-5,
    }
    bg1 = gates["tree_nll_better_than_global"] and gates["tree_nll_better_than_random"]
    bg2 = bg1 and all(gates[name] for name in (
        "simlex_covered_pairs_ge_300", "tree_simlex_rho_ge_0_12",
        "tree_minus_random_rho_ge_0_05", "tree_minus_frequency_rho_ge_0_03",
    ))
    bg3 = bg2 and all(gates[name] for name in (
        "tree_sgns_retention_ge_0_35", "tree_stability_better_than_random",
        "mean_utilization_ge_0_75", "mean_occupancy_entropy_ge_0_75", "conservation_le_1e_5",
    ))
    level = "BG3" if bg3 else "BG2" if bg2 else "BG1" if bg1 else "BG0"
    summary = {
        "claim": "S1-TH-BACKGROUND-C01",
        "experiment": "P-S1-TH-BACKGROUND01",
        "config": asdict(cfg),
        "corpus": {
            **vocab, "spm_sha256": sha256(cfg.spm_model), "simlex_sha256": sha256(cfg.simlex),
            "count_stats": count_stats,
        },
        "builds": builds,
        "nll": nll,
        "stability": {"tree": tree_stability, "random": random_stability},
        "simlex": simlex,
        "sgns": sgns_meta,
        "sgns_retention": retention,
        "utilization": utilization,
        "occupancy_entropy": entropy,
        "max_conservation": conservation,
        "gates": gates,
        "background_level": level,
        "elapsed_seconds": time.time() - started,
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence / "simlex_rows.json").write_text(json.dumps(simlex_rows, indent=2) + "\n", encoding="utf-8")
    (evidence / "neighbors.json").write_text(
        json.dumps(nearest_examples(pieces, routes[0].to(device), sgns, frequency), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"target_ids": target_ids, "context_ids": context_ids, "pieces": pieces, "route": routes[0], "leaf": leaves[0]},
        evidence / "landmarks.pt",
    )
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    (evidence / "README.md").write_text(
        f"# TreeHeap Background Landmarks\n\nLevel: `{level}`\n\n```json\n"
        + json.dumps({"nll": nll, "stability": summary["stability"], "simlex": simlex, "gates": gates}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"level": level, "nll": nll, "stability": summary["stability"], "simlex": simlex, "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
