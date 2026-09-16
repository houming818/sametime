#!/usr/bin/env python3
"""Derive token embeddings by recursively unfolding unit root mass."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    decoder_blend: float
    balance_weight: float
    neighbor_k: int
    inspect_pieces: str
    seed: int
    device: str


def context_nll(prediction: torch.Tensor, counts: torch.Tensor) -> float:
    prediction = prediction.to(counts.device, counts.dtype).clamp_min(1e-12)
    return float((-(counts * prediction.log()).sum() / counts.sum().clamp_min(1.0)).item())


def force_inspection_targets(
    target_ids: Sequence[int],
    requested: Sequence[str],
    counts: object,
    sp: spm.SentencePieceProcessor,
) -> Tuple[List[int], List[str]]:
    result = list(target_ids)
    forced: List[str] = []
    requested_ids = {int(sp.piece_to_id(piece)) for piece in requested}
    replace_at = len(result) - 1
    for piece in requested:
        token_id = int(sp.piece_to_id(piece))
        if token_id < 4 or int(counts[token_id]) <= 0 or token_id in result:
            continue
        while replace_at >= 0 and result[replace_at] in requested_ids:
            replace_at -= 1
        if replace_at < 0:
            break
        result[replace_at] = token_id
        replace_at -= 1
        forced.append(piece)
    return result, forced


def initial_gate_parameters(
    x: torch.Tensor,
    routes: torch.Tensor,
    depth: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.zeros((2**depth - 1, x.shape[1]), dtype=x.dtype, device=x.device)
    biases = torch.zeros(2**depth - 1, dtype=x.dtype, device=x.device)
    offset = 0
    for level in range(depth):
        for node in range(2**level):
            if level == 0:
                members = torch.ones(len(x), dtype=torch.bool, device=x.device)
            else:
                prefix_weights = 2 ** torch.arange(level - 1, -1, -1, device=x.device)
                prefix = (routes[:, :level] * prefix_weights).sum(dim=1)
                members = prefix == node
            left = members & (routes[:, level] == 0)
            right = members & (routes[:, level] == 1)
            left_center = x[left].mean(dim=0)
            right_center = x[right].mean(dim=0)
            weights[offset + node] = 2.0 * (right_center - left_center)
            biases[offset + node] = left_center.square().sum() - right_center.square().sum()
        offset += 2**level
    return weights, biases


def root_unfold(
    observation: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    depth: int,
    temperature: float,
) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    mass = torch.ones((len(observation), 1), dtype=observation.dtype, device=observation.device)
    levels = [mass]
    offset = 0
    conservation_error = 0.0
    for level in range(depth):
        width = 2**level
        logits = observation @ weights[offset : offset + width].T + biases[offset : offset + width]
        right_gate = torch.sigmoid(logits / temperature)
        children = torch.stack((mass * (1.0 - right_gate), mass * right_gate), dim=2)
        next_mass = children.reshape(len(observation), width * 2)
        conservation_error = max(
            conservation_error,
            float((next_mass.sum(dim=1) - mass.sum(dim=1)).abs().max().detach().cpu().item()),
        )
        mass = next_mass
        levels.append(mass)
        offset += width
    return mass, levels, conservation_error


def exact_capacity_projection(probability: torch.Tensor) -> torch.Tensor:
    token_count, leaf_count = probability.shape
    if token_count % leaf_count:
        raise ValueError("target vocabulary must be divisible by leaf count")
    capacity_per_leaf = token_count // leaf_count
    expanded_cost = -probability.detach().cpu().numpy().repeat(capacity_per_leaf, axis=1)
    rows, slots = linear_sum_assignment(expanded_cost)
    leaf = torch.empty(token_count, dtype=torch.long)
    leaf[torch.from_numpy(rows)] = torch.from_numpy(slots // capacity_per_leaf)
    return leaf


def select_hard(probability: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "argmax":
        leaf = probability.argmax(dim=1).detach().cpu()
    elif mode == "exact_capacity":
        leaf = exact_capacity_projection(probability)
    else:
        raise ValueError(mode)
    hard = torch.zeros_like(probability)
    hard.scatter_(1, leaf.to(probability.device)[:, None], 1.0)
    straight_through = probability + (hard - probability).detach()
    return straight_through, leaf


def initial_leaf_probability(
    leaf: torch.Tensor,
    train_counts: torch.Tensor,
    leaf_count: int,
    alpha: float,
    blend: float,
) -> torch.Tensor:
    result = torch.full((leaf_count, train_counts.shape[1]), alpha, dtype=train_counts.dtype)
    result.index_add_(0, leaf.cpu(), train_counts)
    result /= result.sum(dim=1, keepdim=True)
    global_probability = train_counts.sum(dim=0) + alpha
    global_probability /= global_probability.sum()
    return blend * result + (1.0 - blend) * global_probability[None, :]


def route_variance(probability: torch.Tensor) -> float:
    return float(probability.var(dim=0, unbiased=False).mean().detach().cpu().item())


def jensen_shannon(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.clamp_min(1e-12)
    b = b.clamp_min(1e-12)
    middle = 0.5 * (a + b)
    value = 0.5 * (a * (a.log() - middle.log())).sum(dim=1)
    value += 0.5 * (b * (b.log() - middle.log())).sum(dim=1)
    return float(value.mean().detach().cpu().item())


def heldout_observation(test_counts: torch.Tensor, alpha: float, device: str) -> torch.Tensor:
    probability = test_counts.to(device, torch.float32) + alpha
    probability /= probability.sum(dim=1, keepdim=True)
    return probability.sqrt()


def evaluate_arm(
    observation: torch.Tensor,
    test_counts: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor,
    decoder_logits: torch.Tensor,
    cfg: Config,
    mode: str,
    shuffled_observation: torch.Tensor,
) -> Dict[str, object]:
    with torch.no_grad():
        route, levels, conservation = root_unfold(
            observation, weights, biases, cfg.depth, cfg.temperature
        )
        hard, leaf = select_hard(route, mode)
        decoder = decoder_logits.softmax(dim=1)
        prediction = (hard @ decoder).cpu()
        shuffled_route, _levels, shuffled_conservation = root_unfold(
            shuffled_observation, weights, biases, cfg.depth, cfg.temperature
        )
        shuffled_hard, shuffled_leaf = select_hard(shuffled_route, mode)
        shuffled_prediction = (shuffled_hard @ decoder).cpu()
        normal_nll = context_nll(prediction, test_counts)
        shuffled_nll = context_nll(shuffled_prediction, test_counts)
        routes = capacity.routes_from_leaf(leaf.to(observation.device), cfg.depth)
        result = {
            "heldout_nll": normal_nll,
            "shuffled_observation_nll": shuffled_nll,
            "input_causal_nll_delta": shuffled_nll - normal_nll,
            "route_agreement_after_shuffle": float(
                (leaf == shuffled_leaf).to(torch.float32).mean().item()
            ),
            "route_js_after_shuffle": jensen_shannon(route, shuffled_route),
            "root_mass_min": float(levels[0].min().item()),
            "root_mass_max": float(levels[0].max().item()),
            "unfold_conservation_max_abs": max(conservation, shuffled_conservation),
            "route_vector_variance": route_variance(route),
            "mean_top1_route_mass": float(route.max(dim=1).values.mean().item()),
            "normalized_route_entropy": float(
                (-(route * route.clamp_min(1e-12).log()).sum(dim=1).mean() / math.log(route.shape[1])).item()
            ),
            **capacity.leaf_metrics(leaf, cfg.depth),
            **capacity.topology_metrics(
                routes,
                heldout_observation(test_counts, cfg.alpha, cfg.device),
                cfg.neighbor_k,
                cfg.seed,
            ),
        }
    return result


def train_arm(
    name: str,
    mode: str,
    observation: torch.Tensor,
    train_counts: torch.Tensor,
    test_counts: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_biases: torch.Tensor,
    initial_decoder: torch.Tensor,
    shuffled_observation: torch.Tensor,
    cfg: Config,
) -> Tuple[Dict[str, object], List[Dict[str, float]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.nn.Parameter(initial_weights.clone().to(cfg.device, torch.float32))
    biases = torch.nn.Parameter(initial_biases.clone().to(cfg.device, torch.float32))
    decoder_logits = torch.nn.Parameter(initial_decoder.to(cfg.device, torch.float32).clamp_min(1e-12).log())
    optimizer = torch.optim.Adam([weights, biases, decoder_logits], lr=cfg.learning_rate)
    train_float = train_counts.to(cfg.device, torch.float32)
    initial_weight_copy = weights.detach().clone()
    trace: List[Dict[str, float]] = []
    finite_gradients = True
    first_context_nll = 0.0
    first_objective = 0.0
    initial_gate_gradient_l2 = 0.0
    initial_decoder_gradient_l2 = 0.0
    initial_route = torch.empty(0)
    initial_eval: Dict[str, object] = {}

    for step in range(cfg.train_steps + 1):
        route, _levels, conservation = root_unfold(
            observation, weights, biases, cfg.depth, cfg.temperature
        )
        hard, leaf = select_hard(route, mode)
        decoder = decoder_logits.softmax(dim=1)
        prediction = (hard @ decoder).clamp_min(1e-12)
        context_loss = -(train_float * prediction.log()).sum() / train_float.sum().clamp_min(1.0)
        leaf_mass = route.mean(dim=0) * route.shape[1]
        balance_loss = (leaf_mass - 1.0).square().mean()
        objective = context_loss + cfg.balance_weight * balance_loss

        if step == 0:
            first_context_nll = float(context_loss.item())
            first_objective = float(objective.item())
            initial_route = route.detach().cpu()
            gate_gradient = torch.autograd.grad(objective, weights, retain_graph=True)[0]
            decoder_gradient = torch.autograd.grad(objective, decoder_logits, retain_graph=True)[0]
            initial_gate_gradient_l2 = float(gate_gradient.norm().item())
            initial_decoder_gradient_l2 = float(decoder_gradient.norm().item())
            initial_eval = evaluate_arm(
                observation,
                test_counts,
                weights,
                biases,
                decoder_logits,
                cfg,
                mode,
                shuffled_observation,
            )

        if step % 20 == 0 or step == cfg.train_steps:
            occupancy = torch.bincount(leaf, minlength=2**cfg.depth)
            trace.append(
                {
                    "step": step,
                    "train_context_nll": float(context_loss.item()),
                    "balance_loss": float(balance_loss.item()),
                    "objective": float(objective.item()),
                    "conservation_max_abs": conservation,
                    "hard_leaf_min": int(occupancy.min().item()),
                    "hard_leaf_max": int(occupancy.max().item()),
                    "route_vector_variance": route_variance(route),
                }
            )
        if step == cfg.train_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradients = (weights.grad, biases.grad, decoder_logits.grad)
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            finite_gradients = False
            break
        optimizer.step()

    final_eval = evaluate_arm(
        observation,
        test_counts,
        weights,
        biases,
        decoder_logits,
        cfg,
        mode,
        shuffled_observation,
    )
    metrics = {
        "name": name,
        "selector": mode,
        "initial_train_context_nll": first_context_nll,
        "final_train_context_nll": trace[-1]["train_context_nll"],
        "train_nll_decrease": first_context_nll - trace[-1]["train_context_nll"],
        "initial_objective": first_objective,
        "final_objective": trace[-1]["objective"],
        "initial_gate_gradient_l2": initial_gate_gradient_l2,
        "initial_decoder_gradient_l2": initial_decoder_gradient_l2,
        "gate_parameter_delta_l2": float((weights.detach() - initial_weight_copy).norm().item()),
        "finite_gradients": finite_gradients,
        "trainable_token_parameter_count": 0,
        "trainable_gate_parameter_count": int(weights.numel() + biases.numel()),
        "trainable_decoder_parameter_count": int(decoder_logits.numel()),
        "initial": initial_eval,
        "final": final_eval,
    }
    return (
        metrics,
        trace,
        initial_route,
        weights.detach(),
        biases.detach(),
        decoder_logits.detach(),
    )


def examples(
    pieces: Sequence[str],
    initial_route: torch.Tensor,
    final_route: torch.Tensor,
    final_leaf: torch.Tensor,
    requested: Sequence[str],
    limit: int = 10,
) -> List[Dict[str, object]]:
    indices: List[int] = []
    for piece in requested:
        if piece in pieces:
            indices.append(pieces.index(piece))
    for index in range(len(pieces)):
        if index not in indices:
            indices.append(index)
        if len(indices) >= limit:
            break
    return [
        {
            "piece": pieces[index],
            "root": 1.0,
            "initial_leaf_values": [round(float(value), 7) for value in initial_route[index].tolist()],
            "final_leaf_values": [round(float(value), 7) for value in final_route[index].tolist()],
            "selected_leaf": int(final_leaf[index].item()),
        }
        for index in indices[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="S1-ROOT-UNFOLD-EMBED-C01")
    parser.add_argument("--experiment", default="P-S1-ROOT-UNFOLD-EMBED08")
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--max-scan-lines", type=int, default=50_000)
    parser.add_argument("--target-vocab", type=int, default=128)
    parser.add_argument("--context-vocab", type=int, default=256)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-sentence-tokens", type=int, default=96)
    parser.add_argument("--test-mod", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--split-iterations", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--decoder-blend", type=float, default=0.50)
    parser.add_argument("--balance-weight", type=float, default=0.10)
    parser.add_argument("--neighbor-k", type=int, default=3)
    parser.add_argument("--inspect-pieces", default="▁push,▁bank,▁like,▁want")
    parser.add_argument("--seed", type=int, default=19501)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(**vars(parser.parse_args()))
    started = time.time()
    evidence = Path(cfg.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    if cfg.target_vocab % (2**cfg.depth):
        raise ValueError("target-vocab must be divisible by leaf count")

    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model)
    vocab = real.choose_vocab(cfg, sp)
    requested = [piece.strip() for piece in cfg.inspect_pieces.split(",") if piece.strip()]
    target_ids, forced = force_inspection_targets(
        vocab["target_ids"], requested, vocab["counts"], sp
    )
    train, test, corpus = real.build_counts(cfg, sp, target_ids, vocab["context_ids"])
    train_probability = anneal.distributions(train, cfg.alpha)
    test_probability = anneal.distributions(test, cfg.alpha)
    observation = train_probability.sqrt().to(cfg.device, torch.float32)
    heldout_x = test_probability.sqrt().to(cfg.device, torch.float32)

    semantic = capacity.build_balanced_semantic_tree(
        observation, cfg.depth, cfg.split_iterations
    )
    semantic_routes = semantic["routes"]
    semantic_leaf = semantic["leaf"].cpu()
    initial_weights, initial_biases = initial_gate_parameters(
        observation, semantic_routes, cfg.depth
    )
    initial_decoder = initial_leaf_probability(
        semantic_leaf,
        train,
        2**cfg.depth,
        cfg.alpha,
        cfg.decoder_blend,
    )

    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    permutation = torch.randperm(cfg.target_vocab, generator=generator).to(cfg.device)
    shuffled_observation = observation[permutation]
    global_probability = train.sum(dim=0) + cfg.alpha
    global_probability /= global_probability.sum()
    global_nll = context_nll(global_probability[None, :].expand(cfg.target_vocab, -1), test)
    random_control = capacity.balanced_control(
        "random", train.sum(dim=1), cfg.depth, cfg.seed
    )
    random_topology = capacity.topology_metrics(
        random_control["routes"].to(cfg.device), heldout_x, cfg.neighbor_k, cfg.seed
    )

    arms: Dict[str, object] = {}
    traces: Dict[str, object] = {}
    artifacts: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, mode in (("local_argmax", "argmax"), ("capacity_selected", "exact_capacity")):
        metrics, trace, initial_route, weights, biases, decoder_logits = train_arm(
            name,
            mode,
            observation,
            train,
            test,
            initial_weights,
            initial_biases,
            initial_decoder,
            shuffled_observation,
            cfg,
        )
        arms[name] = metrics
        traces[name] = trace
        artifacts[name] = (initial_route, weights, biases, decoder_logits)

    initial_route, final_weights, final_biases, _final_decoder = artifacts["capacity_selected"]
    final_route, level_masses, _conservation = root_unfold(
        observation, final_weights, final_biases, cfg.depth, cfg.temperature
    )
    _hard, final_leaf = select_hard(final_route, "exact_capacity")
    selected_examples = examples(
        [sp.id_to_piece(token_id) for token_id in target_ids],
        initial_route,
        final_route.cpu(),
        final_leaf,
        requested,
    )
    capacity_metrics = arms["capacity_selected"]
    final_metrics = capacity_metrics["final"]
    frozen_metrics = capacity_metrics["initial"]
    expected_capacity = cfg.target_vocab // (2**cfg.depth)
    gates = {
        "unit_root_mass": final_metrics["root_mass_min"] == 1.0
        and final_metrics["root_mass_max"] == 1.0,
        "unfold_mass_conservation": final_metrics["unfold_conservation_max_abs"] < 1e-6,
        "no_trainable_token_parameters": capacity_metrics["trainable_token_parameter_count"] == 0,
        "finite_gradients": capacity_metrics["finite_gradients"],
        "shared_gates_receive_gradient": capacity_metrics["initial_gate_gradient_l2"] > 0.0,
        "shared_gates_changed": capacity_metrics["gate_parameter_delta_l2"] > 1e-6,
        "train_nll_decrease_ge_0_01": capacity_metrics["train_nll_decrease"] >= 0.01,
        "exact_hard_capacity": final_metrics["leaf_count_min"] == expected_capacity
        and final_metrics["leaf_count_max"] == expected_capacity,
        "heldout_beats_global_by_0_05": final_metrics["heldout_nll"] <= global_nll - 0.05,
        "heldout_retains_frozen_unfold": final_metrics["heldout_nll"]
        <= frozen_metrics["heldout_nll"] + 0.03,
        "input_causality_delta_ge_0_02": final_metrics["input_causal_nll_delta"] >= 0.02,
        "topology_beats_balanced_random": final_metrics["heldout_neighbor_lcp"]
        > random_topology["heldout_neighbor_lcp"],
        "derived_embeddings_vary": final_metrics["route_vector_variance"] > 1e-6,
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
            "forced_inspection_pieces": forced,
        },
        "architecture_audit": {
            "root_input": 1.0,
            "level_widths": [2**level for level in range(cfg.depth + 1)],
            "final_embedding_width": 2**cfg.depth,
            "direct_token_leaf_parameter": False,
            "trainable_token_lookup": False,
            "token_condition": "fixed sqrt(P_train(context|token))",
            "unfold_level_mass_ranges": [
                {
                    "level": level,
                    "min_sum": float(mass.sum(dim=1).min().item()),
                    "max_sum": float(mass.sum(dim=1).max().item()),
                }
                for level, mass in enumerate(level_masses)
            ],
        },
        "controls": {
            "global_context_heldout_nll": global_nll,
            "balanced_random_topology": random_topology,
            "semantic_initializer_heldout_nll": capacity.heldout_nll(
                semantic_leaf, train, test, cfg.alpha
            ),
        },
        "arms": arms,
        "examples": selected_examples,
        "gates": gates,
        "smoke_pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (evidence / "trace_argmax.json").write_text(
        json.dumps(traces["local_argmax"], indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "trace_capacity.json").write_text(
        json.dumps(traces["capacity_selected"], indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
