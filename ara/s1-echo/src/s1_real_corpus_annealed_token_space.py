"""Real-WMT audit of annealed FOLD/UNFOLD token coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm
import torch
import torch.nn.functional as F

import s1_annealed_fold_unfold_token_space as anneal


EPS = 1e-12
WORD_PIECE = re.compile(r"^▁[A-Za-z][A-Za-z'-]+$")


@dataclass
class Config:
    data: str
    spm_model: str
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
    alpha: float
    frequency_bin: int
    sgns_dim: int
    sgns_steps: int
    sgns_lr: float
    sgns_negatives: float
    device: str


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def iter_english(path: str, max_lines: int) -> Iterable[Tuple[int, bytes, str]]:
    with open(path, "rb") as handle:
        for line_index, raw in enumerate(handle):
            if line_index >= max_lines:
                break
            try:
                text = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                continue
            fields = text.split("\t")
            if len(fields) < 2 or not fields[1].strip():
                continue
            yield line_index, raw, fields[1]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def choose_vocab(cfg: Config, sp: spm.SentencePieceProcessor) -> Dict[str, object]:
    counts: Counter[int] = Counter()
    scan_digest = hashlib.sha256()
    valid_lines = 0
    token_total = 0
    for _line_index, raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        scan_digest.update(raw)
        ids = [i for i in sp.encode(text, out_type=int) if i >= 4]
        ids = ids[: cfg.max_sentence_tokens]
        counts.update(ids)
        valid_lines += 1
        token_total += len(ids)
    target_candidates = [
        token_id
        for token_id, _count in counts.most_common()
        if WORD_PIECE.fullmatch(sp.id_to_piece(token_id))
    ]
    target_ids = target_candidates[: cfg.target_vocab]
    context_ids = [token_id for token_id, _ in counts.most_common(cfg.context_vocab)]
    if len(target_ids) < cfg.target_vocab:
        raise RuntimeError(f"only {len(target_ids)} readable target pieces found")
    if len(context_ids) < cfg.context_vocab:
        raise RuntimeError(f"only {len(context_ids)} context pieces found")
    return {
        "counts": counts,
        "target_ids": target_ids,
        "context_ids": context_ids,
        "valid_lines": valid_lines,
        "token_total": token_total,
        "scan_sha256": scan_digest.hexdigest(),
    }


def build_counts(
    cfg: Config,
    sp: spm.SentencePieceProcessor,
    target_ids: Sequence[int],
    context_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    target_map = {token_id: i for i, token_id in enumerate(target_ids)}
    context_map = {token_id: i for i, token_id in enumerate(context_ids)}
    train = torch.zeros((len(target_ids), len(context_ids)), dtype=torch.float64)
    test = torch.zeros_like(train)
    stats = {"train_lines": 0, "test_lines": 0, "train_pairs": 0, "test_pairs": 0}
    for line_index, _raw, text in iter_english(cfg.data, cfg.max_scan_lines):
        ids = [i for i in sp.encode(text, out_type=int) if i >= 4]
        ids = ids[: cfg.max_sentence_tokens]
        is_test = line_index % cfg.test_mod == 0
        matrix = test if is_test else train
        line_pairs = 0
        for pos, center_id in enumerate(ids):
            center = target_map.get(center_id)
            if center is None:
                continue
            lo = max(0, pos - cfg.window)
            hi = min(len(ids), pos + cfg.window + 1)
            for other in range(lo, hi):
                if other == pos:
                    continue
                context = context_map.get(ids[other])
                if context is not None:
                    matrix[center, context] += 1.0
                    line_pairs += 1
        key = "test" if is_test else "train"
        stats[f"{key}_lines"] += 1
        stats[f"{key}_pairs"] += line_pairs
    return train, test, stats


def bootstrap_counts(counts: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.poisson(counts, generator=generator)


def frequency_matched_shuffle(counts: torch.Tensor, bin_size: int, seed: int) -> torch.Tensor:
    frequency = counts.sum(dim=1)
    order = torch.argsort(frequency)
    permutation = torch.arange(len(order))
    rng = random.Random(seed)
    for start in range(0, len(order), bin_size):
        group = order[start : start + bin_size]
        if len(group) <= 1:
            continue
        shift = rng.randrange(1, len(group))
        permutation[group] = torch.roll(group, shifts=shift)
    return counts[permutation]


def flat_kmeans(x: torch.Tensor, clusters: int, iterations: int = 40) -> torch.Tensor:
    mean = x.mean(dim=0)
    first = int(((x - mean) ** 2).sum(dim=1).argmax().item())
    chosen = [first]
    min_distance = ((x - x[first]) ** 2).sum(dim=1)
    for _ in range(1, clusters):
        candidate = int(min_distance.argmax().item())
        chosen.append(candidate)
        distance = ((x - x[candidate]) ** 2).sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)
    centers = x[torch.tensor(chosen, device=x.device)].clone()
    assignment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(iterations):
        distance = torch.cdist(x, centers) ** 2
        new_assignment = distance.argmin(dim=1)
        if torch.equal(new_assignment, assignment):
            break
        assignment = new_assignment
        nearest_distance = distance.min(dim=1).values
        for cluster in range(clusters):
            members = assignment == cluster
            if members.any():
                centers[cluster] = x[members].mean(dim=0)
            else:
                replacement = int(nearest_distance.argmax().item())
                centers[cluster] = x[replacement]
                assignment[replacement] = cluster
                nearest_distance[replacement] = -1.0
    return assignment


def expected_sgns(
    counts: torch.Tensor,
    cfg: Config,
    seed: int,
) -> Tuple[torch.Tensor, float, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data = counts.to(cfg.device, dtype=torch.float32)
    n_tokens, n_contexts = data.shape
    center = torch.nn.Parameter(torch.randn(n_tokens, cfg.sgns_dim, device=cfg.device) * 0.02)
    context = torch.nn.Parameter(torch.randn(n_contexts, cfg.sgns_dim, device=cfg.device) * 0.02)
    optimizer = torch.optim.AdamW([center, context], lr=cfg.sgns_lr)
    context_count = data.sum(dim=0).clamp_min(1.0)
    noise = context_count.pow(0.75)
    noise /= noise.sum()
    row_mass = data.sum(dim=1).clamp_min(1.0)
    trace: List[Dict[str, float]] = []
    for step in range(cfg.sgns_steps):
        score = center @ context.t()
        negative = cfg.sgns_negatives * row_mass[:, None] * noise[None, :]
        loss = -(
            data * F.logsigmoid(score) + negative * F.logsigmoid(-score)
        ).sum() / data.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, cfg.sgns_steps - 1) or (step + 1) % 100 == 0:
            trace.append({"step": step + 1, "loss": float(loss.detach().item())})
    with torch.no_grad():
        score = center @ context.t()
        reconstructed_logits = score + noise.clamp_min(EPS).log()[None, :]
        probability = reconstructed_logits.softmax(dim=1).to(torch.float64)
    return F.normalize(center.detach(), dim=1).to(torch.float64), probability, trace


def matrix_nll(probability: torch.Tensor, test_counts: torch.Tensor) -> float:
    probability = probability.to(test_counts.device, dtype=torch.float64).clamp_min(EPS)
    return float((-(test_counts * probability.log()).sum() / test_counts.sum()).item())


def partitions_from_assignment(assignment: torch.Tensor) -> List[int]:
    return assignment.detach().cpu().tolist()


def partition_stability(partitions: Sequence[Sequence[int]]) -> float:
    values = [
        anneal.co_cluster_f1(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    ]
    return statistics.fmean(values) if values else 1.0


def topk_neighbors(coordinates: torch.Tensor, k: int) -> torch.Tensor:
    distance = torch.cdist(coordinates, coordinates)
    distance.fill_diagonal_(float("inf"))
    return distance.topk(k=k, largest=False).indices


def neighbor_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    values = []
    for row_a, row_b in zip(a.cpu().tolist(), b.cpu().tolist()):
        values.append(len(set(row_a) & set(row_b)) / len(row_a))
    return statistics.fmean(values)


def frequency_r2(assignment: torch.Tensor, frequency: torch.Tensor) -> float:
    y = frequency.clamp_min(1.0).log()
    mean = y.mean()
    total = ((y - mean) ** 2).sum().clamp_min(EPS)
    predicted = torch.empty_like(y)
    for cluster in torch.unique(assignment):
        members = assignment == cluster
        predicted[members] = y[members].mean()
    return float((1.0 - ((y - predicted) ** 2).sum() / total).item())


def readable_neighbors(
    pieces: Sequence[str],
    tree_neighbors: torch.Tensor,
    flat_neighbors: torch.Tensor,
    sgns_neighbors: torch.Tensor,
    frequency: torch.Tensor,
    limit: int = 16,
) -> List[Dict[str, object]]:
    ranks = list(range(min(8, len(pieces))))
    for wanted in ("▁push", "▁apple", "▁water", "▁government", "▁people", "▁school", "▁work", "▁time"):
        if wanted in pieces:
            ranks.append(pieces.index(wanted))
    seen = set()
    rows = []
    for token in ranks:
        if token in seen or len(rows) >= limit:
            continue
        seen.add(token)
        rows.append(
            {
                "token": pieces[token],
                "train_frequency": int(frequency[token].item()),
                "tree": [pieces[i] for i in tree_neighbors[token].cpu().tolist()],
                "flat": [pieces[i] for i in flat_neighbors[token].cpu().tolist()],
                "sgns": [pieces[i] for i in sgns_neighbors[token].cpu().tolist()],
            }
        )
    return rows


def evaluate_seed(
    train: torch.Tensor,
    test: torch.Tensor,
    cfg: Config,
    seed: int,
    pieces: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    device = torch.device(cfg.device)
    observed_cpu = bootstrap_counts(train, seed)
    observed = observed_cpu.to(device)
    test_device = test.to(device)
    x = anneal.distributions(observed, cfg.alpha).sqrt()
    weights = observed.sum(dim=1)

    tree = anneal.build_tree(x, weights, cfg.depth, cfg.temperatures, cfg.em_steps)
    shuffled = frequency_matched_shuffle(observed_cpu, cfg.frequency_bin, seed + 100_000).to(device)
    shuffled_tree = anneal.build_tree(
        anneal.distributions(shuffled, cfg.alpha).sqrt(),
        shuffled.sum(dim=1),
        cfg.depth,
        cfg.temperatures,
        cfg.em_steps,
    )
    random_tree = anneal.random_tree(len(pieces), cfg.depth, seed + 200_000, device)
    flat_assignment = flat_kmeans(x, 2**cfg.depth)
    sgns_coordinate, sgns_probability, sgns_trace = expected_sgns(observed_cpu, cfg, seed + 300_000)
    sgns_coordinate = sgns_coordinate.to(device)

    tree_leaf = tree["leaf_ids"]
    shuffled_leaf = shuffled_tree["leaf_ids"]
    random_leaf = random_tree["leaf_ids"]
    tree_util, tree_entropy = anneal.leaf_metrics(tree_leaf, cfg.depth)
    shuffle_util, shuffle_entropy = anneal.leaf_metrics(shuffled_leaf, cfg.depth)
    random_util, random_entropy = anneal.leaf_metrics(random_leaf, cfg.depth)
    flat_util, flat_entropy = anneal.leaf_metrics(flat_assignment, cfg.depth)
    token_nll, global_nll = anneal.reference_nll(observed, test_device, cfg.alpha)

    tree_neighbors = topk_neighbors(tree["route_probs"], 10)
    random_neighbors = topk_neighbors(random_tree["route_probs"].to(torch.float64), 10)
    flat_onehot = F.one_hot(flat_assignment, num_classes=2**cfg.depth).to(torch.float64)
    flat_neighbors = topk_neighbors(flat_onehot, 10)
    sgns_neighbors = topk_neighbors(sgns_coordinate, 10)

    metrics = {
        "seed": seed,
        "nll": {
            "tree": anneal.leaf_context_nll(observed, test_device, tree_leaf, cfg.alpha),
            "frequency_shuffle": anneal.leaf_context_nll(shuffled, test_device, shuffled_leaf, cfg.alpha),
            "random_route": anneal.leaf_context_nll(observed, test_device, random_leaf, cfg.alpha),
            "flat_kmeans": anneal.leaf_context_nll(observed, test_device, flat_assignment, cfg.alpha),
            "expected_sgns": matrix_nll(sgns_probability.to(device), test_device),
            "token_reference": token_nll,
            "global_reference": global_nll,
        },
        "leaf_utilization": {
            "tree": tree_util,
            "frequency_shuffle": shuffle_util,
            "random_route": random_util,
            "flat_kmeans": flat_util,
        },
        "occupancy_entropy": {
            "tree": tree_entropy,
            "frequency_shuffle": shuffle_entropy,
            "random_route": random_entropy,
            "flat_kmeans": flat_entropy,
        },
        "fold_conservation_max_abs": float(tree["conservation_max_abs"]),
        "neighbor_overlap": {
            "tree_vs_sgns": neighbor_overlap(tree_neighbors, sgns_neighbors),
            "random_vs_sgns": neighbor_overlap(random_neighbors, sgns_neighbors),
            "flat_vs_sgns": neighbor_overlap(flat_neighbors, sgns_neighbors),
        },
        "frequency_r2": {
            "tree": frequency_r2(tree_leaf, weights),
            "frequency_shuffle": frequency_r2(shuffled_leaf, weights),
            "random_route": frequency_r2(random_leaf, weights),
            "flat_kmeans": frequency_r2(flat_assignment, weights),
        },
        "partitions": {
            "tree": tree_leaf.cpu().tolist(),
            "frequency_shuffle": shuffled_leaf.cpu().tolist(),
            "random_route": random_leaf.cpu().tolist(),
            "flat_kmeans": flat_assignment.cpu().tolist(),
        },
        "sgns_trace": sgns_trace,
    }
    examples = {
        "seed": seed,
        "neighbors": readable_neighbors(
            pieces, tree_neighbors, flat_neighbors, sgns_neighbors, weights.cpu()
        ),
    }
    return metrics, examples


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    nll_names = [
        "tree", "frequency_shuffle", "random_route", "flat_kmeans",
        "expected_sgns", "token_reference", "global_reference",
    ]
    cluster_names = ["tree", "frequency_shuffle", "random_route", "flat_kmeans"]
    summary: Dict[str, object] = {
        "nll": {
            name: mean_std([float(row["nll"][name]) for row in rows])  # type: ignore[index]
            for name in nll_names
        },
        "leaf_utilization": {
            name: mean_std([float(row["leaf_utilization"][name]) for row in rows])  # type: ignore[index]
            for name in cluster_names
        },
        "occupancy_entropy": {
            name: mean_std([float(row["occupancy_entropy"][name]) for row in rows])  # type: ignore[index]
            for name in cluster_names
        },
        "fold_conservation_max_abs": mean_std(
            [float(row["fold_conservation_max_abs"]) for row in rows]
        ),
        "neighbor_overlap": {
            name: mean_std([float(row["neighbor_overlap"][name]) for row in rows])  # type: ignore[index]
            for name in ("tree_vs_sgns", "random_vs_sgns", "flat_vs_sgns")
        },
        "frequency_r2": {
            name: mean_std([float(row["frequency_r2"][name]) for row in rows])  # type: ignore[index]
            for name in cluster_names
        },
    }
    for name in cluster_names:
        partitions = [row["partitions"][name] for row in rows]  # type: ignore[index]
        summary.setdefault("stability", {})[name] = partition_stability(partitions)  # type: ignore[index]
    nll = summary["nll"]
    stability = summary["stability"]
    overlap = summary["neighbor_overlap"]
    utilization = summary["leaf_utilization"]
    occupancy = summary["occupancy_entropy"]
    conservation = summary["fold_conservation_max_abs"]
    gates = {
        "tree_nll_better_than_global": nll["tree"]["mean"] < nll["global_reference"]["mean"],
        "tree_nll_better_than_frequency_shuffle": nll["tree"]["mean"] < nll["frequency_shuffle"]["mean"],
        "tree_nll_better_than_random_route": nll["tree"]["mean"] < nll["random_route"]["mean"],
        "tree_stability_better_than_frequency_shuffle": stability["tree"] > stability["frequency_shuffle"],
        "tree_stability_better_than_random_route": stability["tree"] > stability["random_route"],
        "tree_sgns_overlap_gap_ge_0_02": overlap["tree_vs_sgns"]["mean"] - overlap["random_vs_sgns"]["mean"] >= 0.02,
        "tree_flat_nll_gap_le_0_15": nll["tree"]["mean"] - nll["flat_kmeans"]["mean"] <= 0.15,
        "tree_leaf_utilization_ge_0_75": utilization["tree"]["mean"] >= 0.75,
        "tree_occupancy_entropy_ge_0_75": occupancy["tree"]["mean"] >= 0.75,
        "fold_conservation_le_1e_6": conservation["max"] <= 1e-6,
    }
    summary["gaps"] = {
        "nll_vs_global": nll["global_reference"]["mean"] - nll["tree"]["mean"],
        "nll_vs_frequency_shuffle": nll["frequency_shuffle"]["mean"] - nll["tree"]["mean"],
        "nll_vs_random_route": nll["random_route"]["mean"] - nll["tree"]["mean"],
        "tree_minus_flat_nll": nll["tree"]["mean"] - nll["flat_kmeans"]["mean"],
        "stability_vs_shuffle": stability["tree"] - stability["frequency_shuffle"],
        "stability_vs_random": stability["tree"] - stability["random_route"],
        "sgns_overlap_vs_random": overlap["tree_vs_sgns"]["mean"] - overlap["random_vs_sgns"]["mean"],
    }
    summary["gates"] = gates
    summary["claim_supported"] = all(gates.values())
    return summary


def write_readme(path: Path, payload: Dict[str, object]) -> None:
    summary = payload["summary"]
    nll = summary["nll"]
    lines = [
        "# Real-Corpus Annealed Token Space",
        "",
        "Claim: `S1-ANNEAL-SPACE-REAL-C01`",
        "",
        "## Mean held-out context NLL",
        "",
    ]
    for name, values in nll.items():
        lines.append(f"- `{name}`: `{values['mean']:.6f} +/- {values['std']:.6f}`")
    lines.extend(["", "## Gates", ""])
    for name, passed in summary["gates"].items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")
    lines.extend(
        [
            "",
            f"Decision: `{'supported controlled real-corpus pilot' if summary['claim_supported'] else 'mixed or not supported'}`",
            "",
            "This is a corpus-structure audit, not a generation or translation result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    ap.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    ap.add_argument("--evidence-dir", default="ara/s1-echo/evidence/s1_real_corpus_annealed_token_space")
    ap.add_argument("--max-scan-lines", type=int, default=200_000)
    ap.add_argument("--target-vocab", type=int, default=512)
    ap.add_argument("--context-vocab", type=int, default=1024)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--max-sentence-tokens", type=int, default=96)
    ap.add_argument("--test-mod", type=int, default=10)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--bootstrap-seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=19101)
    ap.add_argument("--temperatures", default="4,2,1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--em-steps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--frequency-bin", type=int, default=16)
    ap.add_argument("--sgns-dim", type=int, default=32)
    ap.add_argument("--sgns-steps", type=int, default=300)
    ap.add_argument("--sgns-lr", type=float, default=0.03)
    ap.add_argument("--sgns-negatives", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = Config(
        data=args.data,
        spm_model=args.spm_model,
        max_scan_lines=args.max_scan_lines,
        target_vocab=args.target_vocab,
        context_vocab=args.context_vocab,
        window=args.window,
        max_sentence_tokens=args.max_sentence_tokens,
        test_mod=args.test_mod,
        depth=args.depth,
        bootstrap_seeds=args.bootstrap_seeds,
        seed_start=args.seed_start,
        temperatures=[float(v) for v in args.temperatures.split(",")],
        em_steps=args.em_steps,
        alpha=args.alpha,
        frequency_bin=args.frequency_bin,
        sgns_dim=args.sgns_dim,
        sgns_steps=args.sgns_steps,
        sgns_lr=args.sgns_lr,
        sgns_negatives=args.sgns_negatives,
        device=args.device,
    )
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    vocab = choose_vocab(cfg, sp)
    target_ids = vocab.pop("target_ids")
    context_ids = vocab.pop("context_ids")
    corpus_counts = vocab.pop("counts")
    pieces = [sp.id_to_piece(token_id) for token_id in target_ids]
    train, test, count_stats = build_counts(cfg, sp, target_ids, context_ids)
    if (train.sum(dim=1) == 0).any() or (test.sum(dim=1) == 0).any():
        raise RuntimeError("target token with empty train or test context row")
    torch.save(
        {"train": train, "test": test, "target_ids": target_ids, "context_ids": context_ids},
        evidence / "context_counts.pt",
    )
    vocab_payload = {
        **vocab,
        "spm_sha256": file_sha256(cfg.spm_model),
        "target_ids": target_ids,
        "target_pieces": pieces,
        "target_raw_frequency": [int(corpus_counts[token_id]) for token_id in target_ids],
        "context_ids": context_ids,
        "context_pieces": [sp.id_to_piece(token_id) for token_id in context_ids],
        "count_stats": count_stats,
    }
    (evidence / "vocab.json").write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows: List[Dict[str, object]] = []
    examples: List[Dict[str, object]] = []
    with (evidence / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for offset in range(cfg.bootstrap_seeds):
            seed = cfg.seed_start + offset
            metrics, readable = evaluate_seed(train, test, cfg, seed, pieces)
            rows.append(metrics)
            if offset == 0:
                examples.append(readable)
            handle.write(json.dumps(metrics, ensure_ascii=True) + "\n")
            handle.flush()
            print(
                f"seed={seed} tree_nll={metrics['nll']['tree']:.6f} "
                f"flat_nll={metrics['nll']['flat_kmeans']:.6f} "
                f"shuffle_nll={metrics['nll']['frequency_shuffle']:.6f} "
                f"overlap={metrics['neighbor_overlap']['tree_vs_sgns']:.4f}",
                flush=True,
            )
    result_summary = summarize(rows)
    payload = {
        "claim": "S1-ANNEAL-SPACE-REAL-C01",
        "experiment": "P-S1-ANNEAL-SPACE-REAL01",
        "config": asdict(cfg),
        "corpus": vocab_payload,
        "elapsed_seconds": time.time() - started,
        "summary": result_summary,
    }
    (evidence / "results.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "neighbors.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_readme(evidence / "README.md", payload)
    print(json.dumps(result_summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
