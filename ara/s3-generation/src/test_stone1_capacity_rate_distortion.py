#!/usr/bin/env python3
"""Contract tests for the STONE-1 C03 capacity audit."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_capacity_rate_distortion as capacity
import s3_stone1_canonical_codec as c02


def test_registered_parameter_counts() -> None:
    vocab = 32_001
    for arm, config in capacity.ARM_CONFIGS.items():
        formula = capacity.parameter_count(vocab, config["dim"], config["hidden"])
        assert formula == capacity.EXPECTED_PARAMETERS[arm]
        model = c02.make_model(
            "canonical_learned",
            type("Args", (), {
                "dim": config["dim"], "hidden": config["hidden"],
                "heap_width": 64, "leaf_cut": 1,
            })(),
            vocab, 32_000,
        )
        actual = sum(parameter.numel() for parameter in model.parameters())
        assert actual == formula


def test_compute_match_contract() -> None:
    long_budget = capacity.EXPECTED_PARAMETERS["base_28m_long"] * 31_250
    large_budget = capacity.EXPECTED_PARAMETERS["balanced_50m_equal"] * 15_625
    ratio = long_budget / large_budget
    assert 1.09 < ratio < 1.11


def main() -> None:
    tests = [test_registered_parameter_counts, test_compute_match_contract]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
