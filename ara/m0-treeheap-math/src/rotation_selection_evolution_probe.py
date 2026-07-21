#!/usr/bin/env python3
"""Select fixed-capacity rotations using task fitness, not an order label."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import random
import sys
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Candidate:
    name: str
    kind: str
    permutation: torch.Tensor
    edge_preservation: float


class LocalPredictor(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * state_dim + 3, state_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def heap_depth(capacity: int) -> int:
    depth = int(round(math.log2(capacity + 1))) - 1
    if (1 << (depth + 1)) - 1 != capacity:
        raise ValueError("capacity must be 2^(depth+1)-1")
    return depth


def level_bounds(depth: int) -> tuple[int, int]:
    return (1 << depth) - 1, (1 << (depth + 1)) - 1


def parent_index(index: int) -> int:
    return 0 if index == 0 else (index - 1) // 2


def exact_automorphism(capacity: int, rng: random.Random) -> list[int]:
    max_depth = heap_depth(capacity)
    permutation = list(range(capacity))

    def assign(output_index: int, source_index: int, depth: int) -> None:
        permutation[output_index] = source_index
        if depth == max_depth:
            return
        output_left, output_right = 2 * output_index + 1, 2 * output_index + 2
        source_left, source_right = 2 * source_index + 1, 2 * source_index + 2
        if rng.random() < 0.5:
            source_left, source_right = source_right, source_left
        assign(output_left, source_left, depth + 1)
        assign(output_right, source_right, depth + 1)

    assign(0, 0, 0)
    return permutation


def corrupt_within_depth(
    permutation: list[int], swaps: int, capacity: int, rng: random.Random
) -> list[int]:
    result = permutation.copy()
    max_depth = heap_depth(capacity)
    for _ in range(swaps):
        depth = rng.randint(1, max_depth)
        start, end = level_bounds(depth)
        left, right = rng.sample(range(start, end), 2)
        result[left], result[right] = result[right], result[left]
    return result


def random_depth_permutation(capacity: int, rng: random.Random) -> list[int]:
    result = list(range(capacity))
    for depth in range(1, heap_depth(capacity) + 1):
        start, end = level_bounds(depth)
        sources = list(range(start, end))
        rng.shuffle(sources)
        result[start:end] = sources
    return result


def edge_preservation(permutation: list[int]) -> float:
    preserved = 0
    for output_child in range(1, len(permutation)):
        output_parent = parent_index(output_child)
        source_parent = permutation[output_parent]
        source_child = permutation[output_child]
        preserved += int(parent_index(source_child) == source_parent)
    return preserved / (len(permutation) - 1)


def build_candidates(capacity: int, count: int, seed: int) -> list[Candidate]:
    if count != 24:
        raise ValueError("registered population uses 24 candidates")
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    def add(name: str, kind: str, values: list[int]) -> None:
        key = tuple(values)
        if key in seen:
            raise ValueError(f"duplicate candidate {name}")
        seen.add(key)
        candidates.append(
            Candidate(
                name=name,
                kind=kind,
                permutation=torch.tensor(values, dtype=torch.long),
                edge_preservation=edge_preservation(values),
            )
        )

    exact_permutations: list[list[int]] = []
    while len(exact_permutations) < 6:
        values = exact_automorphism(capacity, rng)
        if values != list(range(capacity)) and tuple(values) not in seen:
            add(f"exact_{len(exact_permutations)}", "exact", values)
            exact_permutations.append(values)

    severities = [1, 2, 4, 8, 12, 16]
    for index, swaps in enumerate(severities):
        while True:
            values = corrupt_within_depth(
                exact_permutations[index], swaps, capacity, rng
            )
            if tuple(values) not in seen:
                add(f"mild_{swaps}", "mild", values)
                break

    for index in range(12):
        while True:
            values = random_depth_permutation(capacity, rng)
            if tuple(values) not in seen:
                add(f"random_{index}", "random", values)
                break
    return candidates


def tree_indices(capacity: int, device: torch.device) -> dict[str, torch.Tensor]:
    parent = []
    sibling = []
    left = []
    right = []
    child_count = []
    for index in range(capacity):
        parent.append(parent_index(index))
        if index == 0:
            sibling.append(0)
        elif index % 2 == 1:
            sibling.append(index + 1)
        else:
            sibling.append(index - 1)
        left_index = 2 * index + 1
        right_index = 2 * index + 2
        left.append(left_index if left_index < capacity else 0)
        right.append(right_index if right_index < capacity else 0)
        child_count.append(2.0 if right_index < capacity else 0.0)
    return {
        "parent": torch.tensor(parent, dtype=torch.long, device=device),
        "sibling": torch.tensor(sibling, dtype=torch.long, device=device),
        "left": torch.tensor(left, dtype=torch.long, device=device),
        "right": torch.tensor(right, dtype=torch.long, device=device),
        "child_count": torch.tensor(
            child_count, dtype=torch.float32, device=device
        ).view(1, 1, capacity, 1),
    }


def generate_world(
    batch_size: int,
    capacity: int,
    state_dim: int,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    noise = torch.randn(batch_size, capacity, state_dim, device=device)
    if rho == 0.0:
        return noise
    state = torch.empty_like(noise)
    state[:, 0] = noise[:, 0]
    sigma = math.sqrt(max(0.0, 1.0 - rho * rho))
    max_depth = heap_depth(capacity)
    for depth in range(1, max_depth + 1):
        start, end = level_bounds(depth)
        indices = torch.arange(start, end, device=device)
        parents = (indices - 1) // 2
        state[:, start:end] = rho * state.index_select(1, parents) + sigma * noise[:, start:end]
    return state


def transform_population(state: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    batch_size, capacity, state_dim = state.shape
    candidate_count = permutations.shape[0]
    expanded = state.unsqueeze(1).expand(-1, candidate_count, -1, -1)
    gather_index = permutations.view(1, candidate_count, capacity, 1).expand(
        batch_size, -1, -1, state_dim
    )
    return torch.gather(expanded, 2, gather_index)


def candidate_losses(
    predictor: LocalPredictor,
    state: torch.Tensor,
    permutations: torch.Tensor,
    indices: dict[str, torch.Tensor],
    mask_rate: float,
) -> torch.Tensor:
    transformed = transform_population(state, permutations)
    batch_size, candidate_count, capacity, _ = transformed.shape
    mask = torch.rand(batch_size, 1, capacity, 1, device=state.device) < mask_rate
    mask[:, :, 0] = False
    observed = transformed * (~mask).to(transformed.dtype)

    parent = observed.index_select(2, indices["parent"])
    sibling = observed.index_select(2, indices["sibling"])
    left = observed.index_select(2, indices["left"])
    right = observed.index_select(2, indices["right"])

    observed_flag = (~mask).to(transformed.dtype)
    parent_flag = observed_flag.index_select(2, indices["parent"])
    sibling_flag = observed_flag.index_select(2, indices["sibling"])
    left_flag = observed_flag.index_select(2, indices["left"])
    right_flag = observed_flag.index_select(2, indices["right"])

    parent = parent * parent_flag
    sibling = sibling * sibling_flag
    children_sum = left * left_flag + right * right_flag
    children_observed = (left_flag + right_flag) * (
        indices["child_count"] > 0
    ).to(transformed.dtype)
    children_mean = children_sum / children_observed.clamp_min(1.0)

    flags = torch.cat(
        (
            parent_flag.expand(-1, candidate_count, -1, -1),
            sibling_flag.expand(-1, candidate_count, -1, -1),
            (children_observed / 2.0).expand(-1, candidate_count, -1, -1),
        ),
        dim=-1,
    )
    features = torch.cat((parent, sibling, children_mean, flags), dim=-1)
    prediction = predictor(features)
    node_mse = (prediction - transformed).square().mean(dim=-1)
    target_mask = mask[..., 0].expand(-1, candidate_count, -1)
    denominator = target_mask.sum(dim=(0, 2)).clamp_min(1)
    return (node_mse * target_mask).sum(dim=(0, 2)) / denominator


def group_stats(
    candidates: list[Candidate], probabilities: torch.Tensor, losses: torch.Tensor
) -> dict:
    result: dict[str, dict[str, float]] = {}
    for kind in ("exact", "mild", "random"):
        selected = [index for index, item in enumerate(candidates) if item.kind == kind]
        idx = torch.tensor(selected, dtype=torch.long, device=probabilities.device)
        result[kind] = {
            "count": len(selected),
            "probability_mass": float(probabilities.index_select(0, idx).sum().item()),
            "mean_validation_mse": float(losses.index_select(0, idx).mean().item()),
            "mean_edge_preservation": sum(
                candidates[index].edge_preservation for index in selected
            )
            / len(selected),
        }
    return result


def pearson(values_x: torch.Tensor, values_y: torch.Tensor) -> float:
    x = values_x - values_x.mean()
    y = values_y - values_y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum())).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def train_world(
    name: str,
    rho: float,
    candidates: list[Candidate],
    permutations: torch.Tensor,
    indices: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict:
    torch.manual_seed(args.seed + seed_offset)
    predictor = LocalPredictor(args.state_dim).to(device)
    gate_logits = nn.Parameter(torch.zeros(len(candidates), device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": predictor.parameters(), "lr": args.decoder_lr},
            {"params": [gate_logits], "lr": args.gate_lr},
        ]
    )
    checkpoints = {0, 49, 149, 299, 599, 999, args.steps - 1}
    trace: list[dict] = []

    for step in range(args.steps):
        state = generate_world(
            args.batch_size,
            args.capacity,
            args.state_dim,
            rho,
            device,
        )
        losses = candidate_losses(
            predictor, state, permutations, indices, args.mask_rate
        )
        probabilities = torch.softmax(gate_logits / args.temperature, dim=0)
        if step < args.warmup_steps:
            loss = losses.mean()
        else:
            loss = torch.sum(probabilities * losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            with torch.no_grad():
                current_probs = torch.softmax(
                    gate_logits / args.temperature, dim=0
                )
                exact_mass = sum(
                    float(current_probs[index].item())
                    for index, item in enumerate(candidates)
                    if item.kind == "exact"
                )
                trace.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.item()),
                        "exact_probability_mass": exact_mass,
                        "gate_entropy": float(
                            -torch.sum(
                                current_probs * torch.log(current_probs + 1e-12)
                            ).item()
                        ),
                    }
                )

    validation_losses = torch.zeros(len(candidates), device=device)
    validation_batches = 8
    with torch.no_grad():
        for _ in range(validation_batches):
            state = generate_world(
                args.validation_batch_size,
                args.capacity,
                args.state_dim,
                rho,
                device,
            )
            validation_losses += candidate_losses(
                predictor, state, permutations, indices, args.mask_rate
            )
        validation_losses /= validation_batches
        probabilities = torch.softmax(gate_logits / args.temperature, dim=0)

    edge_scores = torch.tensor(
        [item.edge_preservation for item in candidates],
        dtype=torch.float32,
        device=device,
    )
    candidate_rows = []
    for index, candidate in enumerate(candidates):
        candidate_rows.append(
            {
                "name": candidate.name,
                "kind": candidate.kind,
                "edge_preservation": candidate.edge_preservation,
                "validation_mse": float(validation_losses[index].item()),
                "gate_probability": float(probabilities[index].item()),
            }
        )
    return {
        "name": name,
        "rho": rho,
        "groups": group_stats(candidates, probabilities, validation_losses),
        "edge_preservation_loss_pearson": pearson(edge_scores, validation_losses),
        "gate_entropy": float(
            -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item()
        ),
        "winner": candidate_rows[int(torch.argmax(probabilities).item())],
        "candidates": candidate_rows,
        "training_trace": trace,
    }


def exact_echo_control(
    candidates: list[Candidate],
    state: torch.Tensor,
    device: torch.device,
) -> dict:
    losses = []
    max_errors = []
    for candidate in candidates:
        permutation = candidate.permutation.to(device)
        inverse = torch.argsort(permutation)
        encoded = state.index_select(1, permutation)
        decoded = encoded.index_select(1, inverse)
        difference = decoded - state
        losses.append(float(difference.square().mean().item()))
        max_errors.append(float(difference.abs().max().item()))
    loss_tensor = torch.tensor(losses)
    return {
        "candidate_count": len(candidates),
        "max_inverse_error": max(max_errors),
        "max_candidate_mse": max(losses),
        "candidate_mse_variance": float(loss_tensor.var(unbiased=False).item()),
        "all_candidates_tied": max(losses) == min(losses),
    }


def run_probe(args: argparse.Namespace) -> dict:
    if args.candidates != 24:
        raise ValueError("registered proof uses 24 candidates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = build_candidates(args.capacity, args.candidates, args.seed)
    permutations = torch.stack(
        [item.permutation for item in candidates], dim=0
    ).to(device)
    indices = tree_indices(args.capacity, device)

    structured = train_world(
        "structured",
        args.rho,
        candidates,
        permutations,
        indices,
        args,
        device,
        1000,
    )
    iid = train_world(
        "iid",
        0.0,
        candidates,
        permutations,
        indices,
        args,
        device,
        2000,
    )

    torch.manual_seed(args.seed + 3000)
    echo_state = generate_world(
        256, args.capacity, args.state_dim, args.rho, device
    )
    echo = exact_echo_control(candidates, echo_state, device)

    structured_exact = structured["groups"]["exact"]
    structured_random = structured["groups"]["random"]
    iid_exact = iid["groups"]["exact"]
    iid_random = iid["groups"]["random"]
    gates = {
        "P1_structured_exact_mass": structured_exact["probability_mass"] >= 0.75,
        "P2_structured_loss_margin": (
            structured_random["mean_validation_mse"]
            - structured_exact["mean_validation_mse"]
            >= 0.10
        ),
        "P3_structure_loss_correlation": (
            structured["edge_preservation_loss_pearson"] <= -0.70
        ),
        "P4_iid_no_exact_concentration": iid_exact["probability_mass"] <= 0.50,
        "P5_iid_loss_tie": abs(
            iid_exact["mean_validation_mse"]
            - iid_random["mean_validation_mse"]
        )
        <= 0.02,
        "P6_echo_inverse_exact": echo["max_inverse_error"] < 1e-12,
        "P7_echo_candidates_tied": echo["candidate_mse_variance"] < 1e-12,
        "P8_capacity_constant": permutations.shape[1] == args.capacity,
    }

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "parent_claim": "M0-ROT-C02",
        "predict": "P-ROT02-B",
        "status_before_run": "design",
        "seed": args.seed,
        "device": str(device),
        "capacity_per_candidate": args.capacity,
        "candidate_population": args.candidates,
        "state_dim": args.state_dim,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "mask_rate": args.mask_rate,
        "rho": args.rho,
        "order_signal_in_loss": False,
        "structured": structured,
        "iid": iid,
        "exact_echo": echo,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "boundary": (
            "Controlled Gaussian tree world; tests selected relational order, "
            "not language semantics or universal evolution."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (out / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for world in (structured, iid):
            for row in world["training_trace"]:
                handle.write(
                    json.dumps(
                        {"kind": "training", "world": world["name"], **row},
                        sort_keys=True,
                    )
                    + "\n"
                )
            for row in world["candidates"]:
                handle.write(
                    json.dumps(
                        {"kind": "candidate", "world": world["name"], **row},
                        sort_keys=True,
                    )
                    + "\n"
                )

    readme = f"""# Rotation Selection Evolution Evidence

Parent claim: `M0-ROT-C02`
Predict: `P-ROT02-B`

```text
pilot_pass                         = {summary['pilot_pass']}
device                             = {device}
capacity per candidate             = {args.capacity}
candidate population               = {args.candidates}
order signal in loss               = false

structured exact gate mass         = {structured_exact['probability_mass']:.6f}
structured exact/random MSE        = {structured_exact['mean_validation_mse']:.6f} / {structured_random['mean_validation_mse']:.6f}
structured edge/loss Pearson       = {structured['edge_preservation_loss_pearson']:.6f}

IID exact gate mass                = {iid_exact['probability_mass']:.6f}
IID exact/random MSE               = {iid_exact['mean_validation_mse']:.6f} / {iid_random['mean_validation_mse']:.6f}

exact echo max inverse error       = {echo['max_inverse_error']:.6g}
exact echo candidate loss variance = {echo['candidate_mse_variance']:.6g}
```

Order metrics were audited after training and were not part of the loss.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--capacity", type=int, default=127)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=0.92)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--decoder-lr", type=float, default=0.003)
    parser.add_argument("--gate-lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
