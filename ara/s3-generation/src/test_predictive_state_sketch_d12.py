#!/usr/bin/env python3
"""CPU contracts for the D12 fixed predictive-state sketch."""
from __future__ import annotations

import torch

import s3_predictive_state_sketch_d12 as d12


def main() -> None:
    torch.manual_seed(7)
    pad, vocab, dim, width = 19, 20, 12, 32
    sketch_a = d12.FixedOutcomeSketch(vocab, width, 8, 91)
    sketch_b = d12.FixedOutcomeSketch(vocab, width, 8, 91)
    assert sketch_a.digest() == sketch_b.digest()
    tokens = torch.tensor([[1, 2, 3, 4, pad], [4, 3, 2, 1, pad]])
    encoded = sketch_a(tokens, pad)
    assert torch.allclose(encoded, sketch_b(tokens, pad))
    assert not torch.allclose(encoded[0], encoded[1]), "position binding must preserve order"
    extended_pad = torch.cat((tokens, torch.full((2, 2), pad, dtype=torch.long)), dim=1)
    assert torch.allclose(encoded, sketch_a(extended_pad, pad))

    levels = [
        torch.randn(2, 1, dim, requires_grad=True),
        torch.randn(2, 2, dim, requires_grad=True),
        torch.randn(2, 4, dim, requires_grad=True),
    ]
    masks = [
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
    ]
    predictor = d12.LevelPredictor(dim, 3, width)
    predictions, _ = predictor(levels, masks, detach_states=False)
    loss = d12.sketch_loss(predictions, encoded)
    gradients = torch.autograd.grad(loss, levels, retain_graph=True)
    assert all(float(gradient.norm()) > 0.0 for gradient in gradients)

    detached_predictions, _ = predictor(levels, masks, detach_states=True)
    detached_loss = d12.sketch_loss(detached_predictions, encoded)
    detached_loss.backward()
    assert all(level.grad is None for level in levels)
    assert all(parameter.grad is not None for parameter in predictor.parameters())
    assert torch.isfinite(loss)
    print("D12 CPU contract tests passed")


if __name__ == "__main__":
    main()
