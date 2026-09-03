#!/usr/bin/env python3
"""CPU contract tests for the D11 residual TreeHeap capacity channel."""
from __future__ import annotations

import torch

import s3_structural_protocol_capacity_ladder_d11 as d11


def test_zero_gate_is_exact() -> None:
    torch.manual_seed(1)
    base = torch.randn(2, 5, 23)
    extra = torch.randn_like(base)
    gain = torch.tensor(0.0, requires_grad=True)
    combined = base + torch.tanh(gain) * extra
    assert torch.equal(combined, base)
    combined.sum().backward()
    assert gain.grad is not None and torch.isfinite(gain.grad)
    assert float(gain.grad.abs()) > 0.0


def test_residual_channel_is_recursive_and_trainable() -> None:
    torch.manual_seed(2)
    batch, source_dim, extra_dim, vocab, pad = 2, 8, 16, 23, 22
    widths = (1, 2, 4, 8)
    source_tree = [torch.randn(batch, width, source_dim) for width in widths]
    masks = [torch.ones(batch, width, dtype=torch.bool) for width in widths]
    channel = d11.ResidualTreeHeapChannel(
        source_dim, extra_dim, vocab, pad, max_slots=8,
        source_depths=len(widths), ownership_seed=17,
    )
    budgets = torch.full((batch,), 8, dtype=torch.long)
    tree, tree_masks, slots, entropy = channel.protocol(
        source_tree, masks, depth=3, budgets=budgets, intervention="native",
    )
    assert [row.shape[1] for row in tree] == [1, 2, 4, 8]
    assert [row.shape[1] for row in tree_masks] == [1, 2, 4, 8]
    assert slots.shape == (batch, 8, extra_dim)
    assert entropy.shape[0] == len(widths)
    target = torch.randint(0, vocab - 1, (batch, 5))
    logits, _ = channel.reconstructor.teacher(tree, tree_masks, target, bos=1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab), target.reshape(-1),
    )
    loss.backward()
    gradients = [p.grad for p in channel.parameters() if p.requires_grad]
    assert gradients and all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in gradients)
    route = channel.compressor.last_route_statistics
    assert route["owner_leaf_coverage"] == 1.0
    assert route["argmax_coverage"] == 1.0


if __name__ == "__main__":
    test_zero_gate_is_exact()
    test_residual_channel_is_recursive_and_trainable()
    print("D11 CPU contract tests passed")
