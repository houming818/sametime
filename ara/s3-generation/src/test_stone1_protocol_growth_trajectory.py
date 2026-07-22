#!/usr/bin/env python3
"""Fast structural tests for STONE-1 C04.

Owner: Nio Log Squad
Author: OpenAI Codex
Created: 2026-07-23
Updated: 2026-07-23
Purpose: Ensure pre-fold mirroring changes root while native fold/unfold closes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_canonical_codec as c02


def main() -> None:
    torch.manual_seed(7)
    encoder = c02.CanonicalLiftingEncoder(32, 8, 8, 31, "canonical_learned")
    source = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    length = torch.tensor([8])
    leaf, root, _, levels, _, _ = encoder.states(source, length)
    assert torch.allclose(leaf, levels[-1], atol=1e-6)
    mirrored_root = encoder.fold(source, length, fold_mirror_depth=0)[1]
    assert not torch.allclose(root, mirrored_root)
    print("PASS prefold mirror and native closure")


if __name__ == "__main__":
    main()
