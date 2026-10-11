#!/usr/bin/env python3
"""Deterministic synthetic checks for the T1 posterior reader."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import s1_multiscale_context_posterior_t1 as t1  # noqa: E402


def main() -> None:
    probability = torch.tensor(
        [[0.85, 0.10, 0.04, 0.01], [0.01, 0.04, 0.10, 0.85]],
        dtype=torch.float64,
    )
    observation = torch.tensor([[0, 0, 0, 0], [3, 3, 3, 3]], dtype=torch.long)
    gold = torch.tensor([0, 1], dtype=torch.long)
    prior = torch.tensor([0.5, 0.5], dtype=torch.float64)
    metrics, event_nll = t1.posterior_metrics(
        probability,
        observation,
        gold,
        prior,
        context_steps=[1, 2, 4],
        batch_size=2,
        keep_nll_at=4,
    )
    assert metrics["4"]["nll"] < metrics["1"]["nll"]
    assert metrics["4"]["entropy"] < metrics["1"]["entropy"]
    assert event_nll is not None and len(event_nll) == 2
    bootstrap = t1.paired_bootstrap_margin(
        event_nll,
        [event_nll + 0.1, event_nll + 0.2],
        seed=1,
        samples=50,
    )
    assert bootstrap["lower_95"] > 0.0
    assert t1.tensor_sha256(gold) == t1.tensor_sha256(gold.clone())
    print({"metrics": metrics, "bootstrap": bootstrap})


if __name__ == "__main__":
    main()
