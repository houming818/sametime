#!/usr/bin/env python3
"""Train an NCE-free context readout under exact balanced assignment mass."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
from scipy.optimize import linear_sum_assignment

import s1_annealed_fold_unfold_token_space as anneal
import s1_balanced_capacity_semantic_tree as capacity
import s1_real_corpus_annealed_token_space as real


@dataclass
class Config:
    claim: str
    experiment: str
    data: str
    spm_model: str
    evidence_dir: str
    max_scan_lines: int
    target_vocab: int
    context_vocab: int
    window: int
    max_sentence_tokens: int
    test_mod: int
    depth: int
    split_iterations: int
    alpha: float
    train_steps: int
    learning_rate: float
    temperature: float
    sinkhorn_steps: int
    route_entropy_weight: float
    assignment_mode: str
    initial_leaf_blend: float
    neighbor_k: int
    seed: int
    device: str


def sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    rows, columns = logits.shape
    log_column_mass = math.log(rows / columns)
    log_q = logits
    for _ in range(iterations):
        log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
        log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) + log_column_mass
    return log_q.exp()


def route_probabilities(
    x: torch.Tensor,
    leaf_probability: torch.Tensor,
    temperature: float,
    balanced: bool,
    sinkhorn_steps: int,
) -> torch.Tensor:
    centers = leaf_probability.clamp_min(1e-12).sqrt()
    scores = -(torch.cdist(x, centers) ** 2) / temperature
    if balanced:
        return sinkhorn(scores, sinkhorn_steps)
    return scores.softmax(dim=1)


def initial_leaf_probability(
    leaf: torch.Tensor,
    train_counts: torch.Tensor,
    leaf_count: int,
    alpha: float,
) -> torch.Tensor:
    result = torch.full((leaf_count, train_counts.shape[1]), alpha, dtype=train_counts.dtype)
    result.index_add_(0, leaf.cpu(), train_counts)
    return result / result.sum(dim=1, keepdim=True)


def context_nll(prediction: torch.Tensor, counts: torch.Tensor) -> float:
    prediction = prediction.to(counts.device, counts.dtype).clamp_min(1e-12)
    return float((-(counts * prediction.log()).sum() / counts.sum().clamp_min(1.0)).item())


def exact_capacity_projection(q: torch.Tensor) -> torch.Tensor:
    token_count, leaf_count = q.shape
    if token_count % leaf_count:
        raise ValueError("token count must be divisible by leaf count")
    capacity_per_leaf = token_count // leaf_count
    expanded_cost = -q.detach().cpu().numpy().repeat(capacity_per_leaf, axis=1)
    rows, slots = linear_sum_assignment(expanded_cost)
    leaf = torch.empty(token_count, dtype=torch.long)
    leaf[torch.from_numpy(rows)] = torch.from_numpy(slots // capacity_per_leaf)
    return leaf


def hard_prediction(leaf: torch.Tensor, leaf_probability: torch.Tensor) -> torch.Tensor:
    return leaf_probability.detach().cpu()[leaf]


def forward_assignment(q: torch.Tensor, balanced: bool, mode: str) -> torch.Tensor:
    if mode == "soft":
        return q
    if mode != "straight_through_hard":
        raise ValueError(f"unknown assignment mode: {mode}")
    leaf = exact_capacity_projection(q) if balanced else q.argmax(dim=1).detach().cpu()
    hard = torch.zeros_like(q)
    hard.scatter_(1, leaf.to(q.device)[:, None], 1.0)
    return q + (hard - q).detach()


def train_arm(
    name: str,
    x: torch.Tensor,
    train_counts: torch.Tensor,
    test_counts: torch.Tensor,
    heldout_x: torch.Tensor,
    initial_probability: torch.Tensor,
    cfg: Config,
    balanced: bool,
) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    logits = torch.nn.Parameter(initial_probability.to(cfg.device, torch.float32).clamp_min(1e-12).log())
    initial_logits = logits.detach().clone()
    optimizer = torch.optim.Adam([logits], lr=cfg.learning_rate)
    train_float = train_counts.to(cfg.device, torch.float32)
    trace: List[Dict[str, float]] = []
    first_nll = None
    first_route_entropy = None
    initial_context_gradient_l2 = 0.0
    initial_entropy_gradient_l2 = 0.0
    finite_gradients = True
    last_grad_norm = 0.0

    for step in range(cfg.train_steps + 1):
        leaf_probability = logits.softmax(dim=1)
        q = route_probabilities(x, leaf_probability, cfg.temperature, balanced, cfg.sinkhorn_steps)
        q_forward = forward_assignment(q, balanced, cfg.assignment_mode)
        prediction = (q_forward @ leaf_probability).clamp_min(1e-12)
        context_loss = -(train_float * prediction.log()).sum() / train_float.sum().clamp_min(1.0)
        route_entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=1).mean()
        objective = context_loss + cfg.route_entropy_weight * route_entropy
        if first_nll is None:
            first_nll = float(context_loss.item())
            first_route_entropy = float(route_entropy.item())
            context_gradient = torch.autograd.grad(context_loss, logits, retain_graph=True)[0]
            entropy_gradient = torch.autograd.grad(route_entropy, logits, retain_graph=True)[0]
            initial_context_gradient_l2 = float(context_gradient.norm().item())
            initial_entropy_gradient_l2 = float(entropy_gradient.norm().item())
        if step % 20 == 0 or step == cfg.train_steps:
            trace.append(
                {
                    "step": step,
                    "train_nll": float(context_loss.item()),
                    "route_entropy": float(route_entropy.item()),
                    "objective": float(objective.item()),
                }
            )
        if step == cfg.train_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            finite_gradients = False
            break
        last_grad_norm = float(logits.grad.norm().item())
        optimizer.step()

    with torch.no_grad():
        leaf_probability = logits.softmax(dim=1)
        q = route_probabilities(x, leaf_probability, cfg.temperature, balanced, cfg.sinkhorn_steps)
        leaf = exact_capacity_projection(q) if balanced else q.argmax(dim=1).cpu()
        routes = capacity.routes_from_leaf(leaf.to(cfg.device), cfg.depth)
        soft_prediction = (q @ leaf_probability).detach().cpu()
        hard_pred = hard_prediction(leaf, leaf_probability)
        row_error = float((q.sum(dim=1) - 1.0).abs().max().item())
        expected_column = len(q) / q.shape[1]
        column_error = float((q.sum(dim=0) - expected_column).abs().max().item())
        assignment_entropy = float((-(q * q.clamp_min(1e-12).log()).sum(dim=1).mean()).item())
        top1_mass = float(q.max(dim=1).values.mean().item())
        top1_leaf = q.argmax(dim=1).cpu()
        metrics = {
            "name": name,
            "balanced": balanced,
            "route_entropy_weight": cfg.route_entropy_weight,
            "assignment_mode": cfg.assignment_mode,
            "initial_train_nll": first_nll,
            "final_train_nll": float(trace[-1]["train_nll"]),
            "train_nll_decrease": float(first_nll - trace[-1]["train_nll"]),
            "initial_route_entropy": first_route_entropy,
            "initial_context_gradient_l2": initial_context_gradient_l2,
            "initial_entropy_gradient_l2": initial_entropy_gradient_l2,
            "gradient_balance_weight": initial_context_gradient_l2 / max(initial_entropy_gradient_l2, 1e-12),
            "final_route_entropy": assignment_entropy,
            "normalized_route_entropy": assignment_entropy / math.log(q.shape[1]),
            "mean_top1_mass": top1_mass,
            "soft_heldout_nll": context_nll(soft_prediction, test_counts),
            "hard_heldout_nll": context_nll(hard_pred, test_counts),
            "hard_soft_nll_gap": context_nll(hard_pred, test_counts) - context_nll(soft_prediction, test_counts),
            "projected_top1_agreement": float((leaf == top1_leaf).to(torch.float32).mean().item()),
            "parameter_delta_l2": float((logits - initial_logits).norm().item()),
            "last_gradient_l2": last_grad_norm,
            "finite_gradients": finite_gradients,
            "soft_row_mass_error": row_error,
            "soft_column_mass_error": column_error,
            **capacity.leaf_metrics(leaf, cfg.depth),
            **capacity.topology_metrics(routes, heldout_x, cfg.neighbor_k, cfg.seed),
        }
    return metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="S1-BALANCED-CAPACITY-TRAIN-C01")
    parser.add_argument("--experiment", default="P-S1-BALANCED-CAPACITY03")
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--max-scan-lines", type=int, default=100_000)
    parser.add_argument("--target-vocab", type=int, default=256)
    parser.add_argument("--context-vocab", type=int, default=512)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-sentence-tokens", type=int, default=96)
    parser.add_argument("--test-mod", type=int, default=10)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--split-iterations", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--sinkhorn-steps", type=int, default=30)
    parser.add_argument("--route-entropy-weight", type=float, default=0.0)
    parser.add_argument("--assignment-mode", choices=("soft", "straight_through_hard"), default="soft")
    parser.add_argument("--initial-leaf-blend", type=float, default=1.0)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19411)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(**vars(parser.parse_args()))
    started = time.time()
    evidence = Path(cfg.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    vocab = real.choose_vocab(cfg, sp)
    train, test, corpus = real.build_counts(cfg, sp, vocab["target_ids"], vocab["context_ids"])
    train_probability = anneal.distributions(train, cfg.alpha)
    test_probability = anneal.distributions(test, cfg.alpha)
    x = train_probability.sqrt().to(cfg.device, torch.float32)
    heldout_x = test_probability.sqrt().to(cfg.device, torch.float32)
    semantic = capacity.build_balanced_semantic_tree(x, cfg.depth, cfg.split_iterations)
    semantic_leaf = semantic["leaf"].cpu()
    leaf_count = 2**cfg.depth
    initial_probability = initial_leaf_probability(semantic_leaf, train, leaf_count, cfg.alpha)
    if not 0.0 <= cfg.initial_leaf_blend <= 1.0:
        raise ValueError("initial-leaf-blend must be in [0, 1]")
    global_probability = (train.sum(dim=0) + cfg.alpha)
    global_probability = global_probability / global_probability.sum()
    initial_probability = (
        cfg.initial_leaf_blend * initial_probability
        + (1.0 - cfg.initial_leaf_blend) * global_probability[None, :]
    )

    random_arm = capacity.balanced_control("random", train.sum(dim=1), cfg.depth, cfg.seed)
    frequency_arm = capacity.balanced_control("frequency", train.sum(dim=1), cfg.depth, cfg.seed)
    frozen_routes = capacity.routes_from_leaf(semantic_leaf.to(cfg.device), cfg.depth)
    frozen = {
        "hard_heldout_nll": capacity.heldout_nll(semantic_leaf, train, test, cfg.alpha),
        **capacity.leaf_metrics(semantic_leaf, cfg.depth),
        **capacity.topology_metrics(frozen_routes, heldout_x, cfg.neighbor_k, cfg.seed),
    }
    initial_model = {
        "hard_heldout_nll": context_nll(initial_probability[semantic_leaf], test),
        "leaf_blend": cfg.initial_leaf_blend,
    }
    controls = {}
    for name, arm in (("random", random_arm), ("frequency", frequency_arm)):
        controls[name] = {
            "hard_heldout_nll": capacity.heldout_nll(arm["leaf"].cpu(), train, test, cfg.alpha),
            **capacity.leaf_metrics(arm["leaf"].cpu(), cfg.depth),
            **capacity.topology_metrics(arm["routes"].to(cfg.device), heldout_x, cfg.neighbor_k, cfg.seed),
        }

    balanced_metrics, balanced_trace = train_arm(
        "sinkhorn_balanced", x, train, test, heldout_x, initial_probability, cfg, True
    )
    unconstrained_metrics, unconstrained_trace = train_arm(
        "row_softmax_unconstrained", x, train, test, heldout_x, initial_probability, cfg, False
    )
    best_control_lcp = max(controls["random"]["heldout_neighbor_lcp"], controls["frequency"]["heldout_neighbor_lcp"])
    gates = {
        "finite_gradients": balanced_metrics["finite_gradients"],
        "train_nll_decrease_ge_0_005": balanced_metrics["train_nll_decrease"] >= 0.005,
        "parameters_changed": balanced_metrics["parameter_delta_l2"] > 0.0,
        "soft_capacity": balanced_metrics["soft_row_mass_error"] < 5e-4
        and balanced_metrics["soft_column_mass_error"] < 5e-4,
        "exact_hard_capacity": balanced_metrics["leaf_count_min"] == cfg.target_vocab // leaf_count
        and balanced_metrics["leaf_count_max"] == cfg.target_vocab // leaf_count,
        "heldout_nll_beats_random": balanced_metrics["hard_heldout_nll"] < controls["random"]["hard_heldout_nll"],
        "heldout_nll_retains_initializer": balanced_metrics["hard_heldout_nll"] <= frozen["hard_heldout_nll"] + 0.03,
        "neighbor_lcp_beats_controls": balanced_metrics["heldout_neighbor_lcp"] > best_control_lcp,
        "neighbor_lcp_retains_initializer": balanced_metrics["heldout_neighbor_lcp"] >= frozen["heldout_neighbor_lcp"] - 0.10,
    }
    summary = {
        "claim": cfg.claim,
        "experiment": cfg.experiment,
        "config": asdict(cfg),
        "corpus": {
            **corpus,
            "valid_lines": vocab["valid_lines"],
            "token_total": vocab["token_total"],
            "scan_sha256": vocab["scan_sha256"],
        },
        "frozen_semantic": frozen,
        "initial_model": initial_model,
        "controls": controls,
        "balanced": balanced_metrics,
        "unconstrained": unconstrained_metrics,
        "gates": gates,
        "smoke_pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence / "trace_balanced.json").write_text(json.dumps(balanced_trace, indent=2) + "\n", encoding="utf-8")
    (evidence / "trace_unconstrained.json").write_text(json.dumps(unconstrained_trace, indent=2) + "\n", encoding="utf-8")
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
