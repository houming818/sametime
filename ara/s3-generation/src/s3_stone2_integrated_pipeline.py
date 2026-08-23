#!/usr/bin/env python3
"""STONE-2 integrated C13 FOLD + no-STOP multi-level READ pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_bounded_annealing_fold_c13_train as c13  # noqa: E402
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-STONE2-INTEGRATED-C03"
_base_builder = c10.build_model


def build_integrated_model(args, vocab: int, pad: int):
    base = _base_builder(args, vocab, pad)
    origin = torch.zeros(args.dim, device=args.device)
    bounded = c13.ReferenceModel(base, origin).to(args.device)
    return c11.HStateConvolutionModel(
        bounded, args.dim, args.hidden, use_up=False,
    ).to(args.device)


def integrated_contract(model, args, vocab: int, pad: int):
    names = [name for name, _ in model.named_parameters()]
    return {
        "claim": CLAIM,
        "class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "vocab": vocab,
        "pad": pad,
        "dim": args.dim,
        "hidden": args.hidden,
        "heap_width": args.heap_width,
        "dynamic_width": True,
        "communication": "xor_butterfly",
        "fold": "reference_normalized_zero_origin",
        "fold_invertible_with_saved_scale": True,
        "read": "mandatory_multilevel_residual",
        "learned_stop_present": any("stop" in name.lower() for name in names),
        "extra_bottom_up_k_up_active": False,
        "raw_leaf_decoder_bypass": False,
        "parameter_memory_topology": "shared_tensor_kernels",
    }


def main():
    c10.CLAIM = CLAIM
    c10.CONTEXT_WIDTHS = (4, 8, 16, 32, 64, 128, 256)
    c10.build_model = build_integrated_model
    c10.model_contract = integrated_contract
    c10.main()


if __name__ == "__main__":
    main()
