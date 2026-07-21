#!/usr/bin/env python3
"""Fixed-capacity nested-rotation private protocol carrier probe.

The encoder applies a hard private program of overlapping subheap mirrors to a
127-node TreeHeap. The decoder has one trainable logit per registered mirror
and learns the inverse program from reconstruction loss only.

This is a protocol-carrier proof. It does not test natural emergence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
from dataclasses import dataclass

import torch
from torch import nn


PROGRAM_A = [1, 1, 0, 1, 0, 1]
PROGRAM_B = [0, 1, 1, 0, 1, 1]
OP_SPECS = [
    ("", 6),
    ("L", 4),
    ("LR", 3),
    ("R", 4),
    ("RL", 3),
    ("LL", 3),
]


@dataclass
class EvalResult:
    mse: float
    max_abs_error: float
    exact_fraction: float


def path_to_index(path: str) -> int:
    index = 0
    for direction in path:
        index = 2 * index + (1 if direction == "L" else 2)
    return index


def local_paths(depth: int) -> list[str]:
    paths = [""]
    frontier = [""]
    for _ in range(depth):
        nxt: list[str] = []
        for path in frontier:
            nxt.extend((path + "L", path + "R"))
        paths.extend(nxt)
        frontier = nxt
    return paths


def mirror_path(path: str) -> str:
    return "".join("R" if ch == "L" else "L" for ch in path)


def mirror_permutation(capacity: int, root_path: str, depth: int) -> torch.Tensor:
    permutation = torch.arange(capacity, dtype=torch.long)
    for local_path in local_paths(depth):
        output_index = path_to_index(root_path + local_path)
        source_index = path_to_index(root_path + mirror_path(local_path))
        if output_index >= capacity or source_index >= capacity:
            raise ValueError(
                f"subheap {root_path!r} depth={depth} exceeds capacity={capacity}"
            )
        permutation[output_index] = source_index
    return permutation


def compose_permutations(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Permutation for applying first, then second, with out[j] = in[perm[j]]."""
    return first.index_select(0, second)


def apply_hard_operation(state: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return state.index_select(1, permutation)


def apply_hard_program(
    state: torch.Tensor,
    permutations: list[torch.Tensor],
    bits: list[int],
    *,
    reverse: bool,
    limit: int | None = None,
) -> torch.Tensor:
    result = state
    order = list(range(len(permutations)))
    if reverse:
        order.reverse()
    if limit is not None:
        order = order[:limit]
    for operation_index in order:
        if bits[operation_index]:
            result = apply_hard_operation(result, permutations[operation_index])
    return result


class SoftInverseDecoder(nn.Module):
    def __init__(self, operation_count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(operation_count))

    def forward(
        self, encoded: torch.Tensor, permutations: list[torch.Tensor]
    ) -> torch.Tensor:
        result = encoded
        probabilities = torch.sigmoid(self.logits)
        for operation_index in reversed(range(len(permutations))):
            mirrored = apply_hard_operation(result, permutations[operation_index])
            probability = probabilities[operation_index]
            result = (1.0 - probability) * result + probability * mirrored
        return result


def make_states(
    samples: int,
    capacity: int,
    state_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    state = torch.randn(samples, capacity, state_dim, generator=generator)
    address = torch.linspace(-0.4, 0.4, capacity).view(1, capacity, 1)
    channel = torch.linspace(0.6, 1.3, state_dim).view(1, 1, state_dim)
    return state + address * channel


def evaluate(prediction: torch.Tensor, target: torch.Tensor) -> EvalResult:
    difference = prediction - target
    per_sample_max = difference.abs().flatten(1).max(dim=1).values
    return EvalResult(
        mse=float(difference.square().mean().item()),
        max_abs_error=float(difference.abs().max().item()),
        exact_fraction=float((per_sample_max < 1e-6).float().mean().item()),
    )


def train_decoder(
    train_state: torch.Tensor,
    train_encoded: torch.Tensor,
    permutations: list[torch.Tensor],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[SoftInverseDecoder, list[dict[str, float]]]:
    device = train_state.device
    model = SoftInverseDecoder(len(permutations)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    index_generator = torch.Generator(device="cpu").manual_seed(seed)
    trace: list[dict[str, float]] = []
    checkpoints = {0, 9, 49, 99, 249, 499, 999, epochs - 1}

    for epoch in range(epochs):
        indices = torch.randint(
            0,
            train_state.shape[0],
            (batch_size,),
            generator=index_generator,
        ).to(device)
        target = train_state.index_select(0, indices)
        encoded = train_encoded.index_select(0, indices)
        prediction = model(encoded, permutations)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch in checkpoints:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "probabilities": [
                        float(value)
                        for value in torch.sigmoid(model.logits).detach().cpu()
                    ],
                }
            )
    return model, trace


def hard_bits(model: SoftInverseDecoder) -> list[int]:
    return [
        int(value >= 0.5)
        for value in torch.sigmoid(model.logits).detach().cpu().tolist()
    ]


def protocol_metrics(
    name: str,
    program: list[int],
    other_program: list[int],
    model: SoftInverseDecoder,
    other_model: SoftInverseDecoder,
    test_state: torch.Tensor,
    permutations: list[torch.Tensor],
) -> dict:
    encoded = apply_hard_program(
        test_state, permutations, program, reverse=False
    )
    learned_program = hard_bits(model)
    other_learned_program = hard_bits(other_model)

    true_inverse = apply_hard_program(
        encoded, permutations, program, reverse=True
    )
    paired_hard = apply_hard_program(
        encoded, permutations, learned_program, reverse=True
    )
    paired_soft = model(encoded, permutations)
    identity = encoded
    cross = apply_hard_program(
        encoded, permutations, other_learned_program, reverse=True
    )

    flipped = learned_program.copy()
    flip_index = next(index for index, bit in enumerate(flipped) if bit == 1)
    flipped[flip_index] = 0
    one_bit_flip = apply_hard_program(
        encoded, permutations, flipped, reverse=True
    )
    wrong_order = apply_hard_program(
        encoded, permutations, learned_program, reverse=False
    )

    partial_curve = []
    for limit in range(len(permutations) + 1):
        partial = apply_hard_program(
            encoded,
            permutations,
            learned_program,
            reverse=True,
            limit=limit,
        )
        metrics = evaluate(partial, test_state)
        partial_curve.append(
            {
                "inverse_steps": limit,
                "mse": metrics.mse,
                "exact_fraction": metrics.exact_fraction,
            }
        )

    return {
        "name": name,
        "encoder_program": program,
        "other_encoder_program": other_program,
        "learned_decoder_program": learned_program,
        "decoder_probabilities": [
            float(value)
            for value in torch.sigmoid(model.logits).detach().cpu().tolist()
        ],
        "bit_accuracy": sum(
            int(expected == actual)
            for expected, actual in zip(program, learned_program)
        )
        / len(program),
        "true_inverse": evaluate(true_inverse, test_state).__dict__,
        "paired_hard": evaluate(paired_hard, test_state).__dict__,
        "paired_soft": evaluate(paired_soft, test_state).__dict__,
        "identity_decoder": evaluate(identity, test_state).__dict__,
        "cross_protocol": evaluate(cross, test_state).__dict__,
        "one_bit_flip": {
            "flipped_index": flip_index,
            **evaluate(one_bit_flip, test_state).__dict__,
        },
        "wrong_inverse_order": evaluate(wrong_order, test_state).__dict__,
        "partial_inverse_curve": partial_curve,
    }


def run_probe(args: argparse.Namespace) -> dict:
    if args.capacity != 127:
        raise ValueError("registered proof uses capacity=127")
    if len(PROGRAM_A) != len(OP_SPECS) or len(PROGRAM_B) != len(OP_SPECS):
        raise AssertionError("program/operator length mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    permutations_cpu = [
        mirror_permutation(args.capacity, root, depth)
        for root, depth in OP_SPECS
    ]
    permutations = [permutation.to(device) for permutation in permutations_cpu]

    involution_exact = all(
        torch.equal(permutation.index_select(0, permutation), torch.arange(args.capacity))
        for permutation in permutations_cpu
    )
    noncommuting_pairs = 0
    for left in range(len(permutations_cpu)):
        for right in range(left + 1, len(permutations_cpu)):
            lr = compose_permutations(permutations_cpu[left], permutations_cpu[right])
            rl = compose_permutations(permutations_cpu[right], permutations_cpu[left])
            noncommuting_pairs += int(not torch.equal(lr, rl))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    train_state = make_states(
        args.train_samples, args.capacity, args.state_dim, generator
    ).to(device)
    test_state = make_states(
        args.test_samples, args.capacity, args.state_dim, generator
    ).to(device)

    train_encoded_a = apply_hard_program(
        train_state, permutations, PROGRAM_A, reverse=False
    )
    train_encoded_b = apply_hard_program(
        train_state, permutations, PROGRAM_B, reverse=False
    )

    model_a, trace_a = train_decoder(
        train_state,
        train_encoded_a,
        permutations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 101,
    )
    model_b, trace_b = train_decoder(
        train_state,
        train_encoded_b,
        permutations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 202,
    )

    result_a = protocol_metrics(
        "A", PROGRAM_A, PROGRAM_B, model_a, model_b, test_state, permutations
    )
    result_b = protocol_metrics(
        "B", PROGRAM_B, PROGRAM_A, model_b, model_a, test_state, permutations
    )

    original_shape = list(test_state.shape)
    encoded_shape = list(
        apply_hard_program(test_state, permutations, PROGRAM_A, reverse=False).shape
    )
    gates = {
        "P1_true_inverse_exact": max(
            result_a["true_inverse"]["max_abs_error"],
            result_b["true_inverse"]["max_abs_error"],
        )
        < 1e-12,
        "P2_decoder_bits_exact": (
            result_a["bit_accuracy"] == 1.0 and result_b["bit_accuracy"] == 1.0
        ),
        "P3_paired_hard_exact": max(
            result_a["paired_hard"]["max_abs_error"],
            result_b["paired_hard"]["max_abs_error"],
        )
        < 1e-6,
        "P4_identity_damaged": min(
            result_a["identity_decoder"]["mse"],
            result_b["identity_decoder"]["mse"],
        )
        > 0.10,
        "P5_cross_protocol_damaged": min(
            result_a["cross_protocol"]["mse"],
            result_b["cross_protocol"]["mse"],
        )
        > 0.10,
        "P6_one_bit_damaged": min(
            result_a["one_bit_flip"]["mse"],
            result_b["one_bit_flip"]["mse"],
        )
        > 0.05,
        "P7_wrong_order_damaged": min(
            result_a["wrong_inverse_order"]["mse"],
            result_b["wrong_inverse_order"]["mse"],
        )
        > 0.05,
        "P8_partial_then_complete": all(
            result["partial_inverse_curve"][-1]["exact_fraction"] == 1.0
            and max(
                row["mse"] for row in result["partial_inverse_curve"][:-1]
            )
            > 0.05
            for result in (result_a, result_b)
        ),
        "P9_capacity_constant": original_shape == encoded_shape,
        "mirror_involution_exact": involution_exact,
        "nested_operations_noncommute": noncommuting_pairs > 0,
    }

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "parent_claim": "M0-ROT-C02",
        "predict": "P-ROT02-A",
        "status_before_run": "design",
        "seed": args.seed,
        "device": str(device),
        "capacity": args.capacity,
        "state_dim": args.state_dim,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "operation_specs": [
            {"root_path": root, "depth": depth} for root, depth in OP_SPECS
        ],
        "noncommuting_operation_pairs": noncommuting_pairs,
        "treeheap_state_shape_before": original_shape,
        "treeheap_state_shape_after_encode": encoded_shape,
        "decoder_parameter_count_each": len(OP_SPECS),
        "protocol_a": result_a,
        "protocol_b": result_b,
        "training_trace_a": trace_a,
        "training_trace_b": trace_b,
        "gates": gates,
        "pilot_pass": all(gates.values()),
        "boundary": (
            "Fixed encoder programs; proves private protocol carriage and learned "
            "inverse decoding, not natural protocol emergence or language value."
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
        for protocol, trace in (("A", trace_a), ("B", trace_b)):
            for row in trace:
                handle.write(
                    json.dumps(
                        {"kind": "training", "protocol": protocol, **row},
                        sort_keys=True,
                    )
                    + "\n"
                )
        for result in (result_a, result_b):
            for row in result["partial_inverse_curve"]:
                handle.write(
                    json.dumps(
                        {
                            "kind": "partial_inverse",
                            "protocol": result["name"],
                            **row,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    readme = f"""# Fixed-Capacity Rotation Protocol Evidence

Parent claim: `M0-ROT-C02`
Predict: `P-ROT02-A`

```text
pilot_pass                  = {summary['pilot_pass']}
device                      = {device}
capacity                    = {args.capacity}
noncommuting pairs          = {noncommuting_pairs}
A learned bits              = {result_a['learned_decoder_program']}
B learned bits              = {result_b['learned_decoder_program']}
A paired/cross MSE          = {result_a['paired_hard']['mse']:.8f} / {result_a['cross_protocol']['mse']:.8f}
B paired/cross MSE          = {result_b['paired_hard']['mse']:.8f} / {result_b['cross_protocol']['mse']:.8f}
A wrong-order/one-bit MSE   = {result_a['wrong_inverse_order']['mse']:.8f} / {result_a['one_bit_flip']['mse']:.8f}
B wrong-order/one-bit MSE   = {result_b['wrong_inverse_order']['mse']:.8f} / {result_b['one_bit_flip']['mse']:.8f}
state shape before/after    = {original_shape} / {encoded_shape}
```

This is a fixed-program carrier proof. It does not show that rotation emerges
without protocol registration or that it improves a language task.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--capacity", type=int, default=127)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
