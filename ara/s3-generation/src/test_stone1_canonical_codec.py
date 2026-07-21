#!/usr/bin/env python3
"""Deterministic algebra, closure, control-matching, and gradient tests for C02."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_canonical_codec as stone


def test_bilateral_algebra() -> None:
    torch.manual_seed(11)
    left = torch.randn(5, 3, 16)
    right = torch.randn(5, 3, 16)
    codec = stone.CanonicalCodec(16, "canonical_algebraic")
    detail = right - codec.predict(left)
    parent = left + codec.update(detail)
    recovered_left = parent - codec.update(detail)
    recovered_right = detail + codec.predict(recovered_left)
    torch.testing.assert_close(detail, right - left, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        parent, 0.4 * left + 0.6 * right, atol=2e-7, rtol=1e-7,
    )
    torch.testing.assert_close(left, recovered_left, atol=2e-7, rtol=1e-7)
    torch.testing.assert_close(right, recovered_right, atol=2e-7, rtol=1e-7)


def test_recursive_closure_and_unpaired_passthrough() -> None:
    torch.manual_seed(12)
    encoder = stone.CanonicalLiftingEncoder(
        vocab=101, dim=16, heap_width=16, pad=100,
        variant="canonical_learned",
    )
    source = torch.randint(0, 100, (4, 13))
    length = torch.tensor([13, 10, 7, 3])
    state = encoder.states(source, length)
    torch.testing.assert_close(state[3][-1], state[0], atol=3e-6, rtol=1e-6)

    # At depth zero, sentence length 13 has an unmatched node at leaf index 12.
    expected = state[0][0, 12]
    folded_parent = encoder.fold(source, length)[3][1]
    assert bool(folded_parent[0, 6])
    recovered = state[3][-2][0, 6]
    torch.testing.assert_close(recovered, expected, atol=1e-7, rtol=1e-7)


def test_algebraic_and_learned_match_at_step_zero() -> None:
    torch.manual_seed(13)
    algebraic = stone.CanonicalLiftingEncoder(101, 16, 16, 100, "canonical_algebraic")
    torch.manual_seed(13)
    learned = stone.CanonicalLiftingEncoder(101, 16, 16, 100, "canonical_learned")
    source = torch.randint(0, 100, (3, 11))
    length = torch.tensor([11, 8, 5])
    a = algebraic.states(source, length)
    b = learned.states(source, length)
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
    for left, right in zip(a[2], b[2]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_translation_signal_reaches_both_residual_outputs() -> None:
    torch.manual_seed(14)
    encoder = stone.CanonicalLiftingEncoder(101, 16, 16, 100, "canonical_learned")
    source = torch.randint(0, 100, (4, 13))
    length = torch.tensor([13, 10, 7, 3])
    state = encoder.states(source, length)
    loss = state[1].square().mean() + sum(row.square().mean() for row in state[2])
    loss.backward()
    predict_grad = encoder.codec.predict_net[-1].weight.grad
    update_grad = encoder.codec.update_net[-1].weight.grad
    assert predict_grad is not None and float(predict_grad.abs().sum()) > 0
    assert update_grad is not None and float(update_grad.abs().sum()) > 0


def test_frozen_codec_does_not_change_later_rng_or_closure() -> None:
    torch.manual_seed(15)
    algebraic = stone.CanonicalLiftingEncoder(101, 16, 16, 100, "canonical_algebraic")
    after_algebraic = torch.randn(8)
    torch.manual_seed(15)
    frozen = stone.CanonicalLiftingEncoder(101, 16, 16, 100, "canonical_frozen")
    after_frozen = torch.randn(8)
    torch.testing.assert_close(after_algebraic, after_frozen, atol=0, rtol=0)
    assert not any(parameter.requires_grad for parameter in frozen.codec.parameters())

    source = torch.randint(0, 100, (4, 13))
    length = torch.tensor([13, 10, 7, 3])
    state = frozen.states(source, length)
    torch.testing.assert_close(state[3][-1], state[0], atol=3e-6, rtol=1e-6)


def main() -> None:
    tests = [
        test_bilateral_algebra,
        test_recursive_closure_and_unpaired_passthrough,
        test_algebraic_and_learned_match_at_step_zero,
        test_translation_signal_reaches_both_residual_outputs,
        test_frozen_codec_does_not_change_later_rng_or_closure,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
