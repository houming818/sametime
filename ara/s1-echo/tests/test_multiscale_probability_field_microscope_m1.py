#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
MODULE = SRC / "s1_multiscale_probability_field_microscope_m1.py"
spec = importlib.util.spec_from_file_location("microscope", MODULE)
microscope = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(microscope)

POSTERIOR_MODULE = (
    SRC / "s1_multiscale_posterior_pyramid_m1b.py"
)
posterior_spec = importlib.util.spec_from_file_location("posterior_microscope", POSTERIOR_MODULE)
posterior_microscope = importlib.util.module_from_spec(posterior_spec)
assert posterior_spec.loader is not None
posterior_spec.loader.exec_module(posterior_microscope)


def test_exact_probability_pyramid() -> None:
    probability = torch.tensor(
        [[0.50, 0.30, 0.15, 0.05], [0.10, 0.20, 0.30, 0.40]],
        dtype=torch.float64,
    )
    coord = microscope.tree_coordinates(probability)
    assert torch.allclose(coord["masses"][0], torch.ones((2, 1), dtype=torch.float64))
    assert torch.allclose(coord["splits"][0][:, 0], torch.tensor([0.8, 0.3], dtype=torch.float64))
    reconstructed = microscope.reconstruct_from_splits(coord["splits"])
    assert torch.allclose(reconstructed, probability, atol=1e-12, rtol=0)


def test_entropy_and_cross_entropy_chain() -> None:
    prediction = torch.tensor(
        [[0.50, 0.30, 0.15, 0.05], [0.10, 0.20, 0.30, 0.40]],
        dtype=torch.float64,
    )
    test_counts = torch.tensor([[5, 3, 1, 1], [1, 2, 3, 4]], dtype=torch.float64)
    prior = torch.tensor([0.6, 0.4], dtype=torch.float64)
    result = microscope.hierarchy_audit(prediction, test_counts, prior, [1, 3])
    assert result["mass_conservation_max_abs"] < 1e-12
    assert result["full_reconstruction_max_abs"] < 1e-12
    assert result["entropy_chain_max_abs"] < 1e-12
    assert result["heldout_ce_chain_max_abs"] < 1e-12


def test_exact_posterior_refinement() -> None:
    fine = torch.tensor(
        [
            [0.50, 0.30, 0.15, 0.05],
            [0.10, 0.20, 0.30, 0.40],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    coarse = fine.reshape(3, 2, 2).sum(dim=2)
    fine_observation = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    coarse_observation = fine_observation // 2
    gold = torch.tensor([0, 1], dtype=torch.long)
    prior = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    result = posterior_microscope.transition_audit(
        coarse,
        fine,
        coarse_observation,
        fine_observation,
        gold,
        prior,
        batch_size=2,
        spectrum_events=2,
    )
    assert result["max_abs_log_closure"] < 1e-6


if __name__ == "__main__":
    test_exact_probability_pyramid()
    test_entropy_and_cross_entropy_chain()
    test_exact_posterior_refinement()
    print("ok")

