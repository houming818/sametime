#!/usr/bin/env python3
"""Run a frozen TreeHeap microscope with structured analytic filter families."""
from __future__ import annotations

import math

import torch

import s3_treeheap_analytic_filter_interference_f06 as f06


CLAIM = "S3-TREEHEAP-STRUCTURED-FILTER-BANK-F07"
FILTER_NAMES = (
    "uniform-gain",
    "pascal-row-energy",
    "gaussian-center-energy",
    "impulse-center-energy",
    "ancestor-chain-energy",
    "subtree-energy",
    "triangle-redistribute",
    "center-surround",
    "alternating-highpass",
    "depth-fine-minus-coarse",
    "haar-local-sibling",
)
ENERGY_FILTERS = frozenset({
    "uniform-gain",
    "pascal-row-energy",
    "gaussian-center-energy",
    "impulse-center-energy",
    "ancestor-chain-energy",
    "subtree-energy",
})
CENTERED_FILTERS = frozenset(FILTER_NAMES) - ENERGY_FILTERS
INTERFERENCE_PAIRS = (
    ("uniform-gain", "gaussian-center-energy"),
    ("gaussian-center-energy", "center-surround"),
    ("ancestor-chain-energy", "haar-local-sibling"),
    ("depth-fine-minus-coarse", "alternating-highpass"),
)


def _target_fine_node(metadata: list[dict]) -> int:
    fine = [meta["node"] for meta in metadata if meta["merge"] == 0]
    if not fine:
        raise RuntimeError("structured filters require an active finest merge")
    return fine[len(fine) // 2]


def _subtree_focus(metadata: list[dict]) -> tuple[int, int]:
    candidates = [meta for meta in metadata if meta["merge"] == 1]
    if not candidates:
        candidates = [meta for meta in metadata if meta["merge"] == 0]
    focus = candidates[len(candidates) // 2]
    return focus["merge"], focus["node"]


def raw_values(name: str, metadata: list[dict]) -> torch.Tensor:
    count = len(metadata)
    center = (count - 1) / 2.0
    sigma = max(count / 5.0, 1.0)
    index = torch.arange(count, dtype=torch.float64)
    offset = index - center

    if name == "uniform-gain":
        return torch.ones(count, dtype=torch.float64)
    if name == "pascal-row-energy":
        values = torch.tensor(
            [math.comb(count - 1, position) for position in range(count)],
            dtype=torch.float64,
        )
        return values / values.max()
    if name == "gaussian-center-energy":
        return torch.exp(-0.5 * (offset / sigma).square())
    if name == "impulse-center-energy":
        values = torch.zeros(count, dtype=torch.float64)
        values[int(center)] = 1.0
        return values
    if name == "ancestor-chain-energy":
        target = _target_fine_node(metadata)
        return torch.tensor([
            float(meta["node"] == target // (2 ** meta["merge"]))
            for meta in metadata
        ], dtype=torch.float64)
    if name == "subtree-energy":
        focus_merge, focus_node = _subtree_focus(metadata)
        return torch.tensor([
            float(
                meta["merge"] <= focus_merge
                and meta["node"] // (2 ** (focus_merge - meta["merge"])) == focus_node
            )
            for meta in metadata
        ], dtype=torch.float64)
    if name == "triangle-redistribute":
        return 1.0 - offset.abs() / max(center, 1.0)
    if name == "center-surround":
        radius = offset / sigma
        return (1.0 - radius.square()) * torch.exp(-0.5 * radius.square())
    if name == "alternating-highpass":
        return torch.tensor(
            [1.0 if position % 2 == 0 else -1.0 for position in range(count)],
            dtype=torch.float64,
        )
    if name == "depth-fine-minus-coarse":
        return torch.tensor(
            [float(meta["root_depth"]) for meta in metadata], dtype=torch.float64,
        )
    if name == "haar-local-sibling":
        return torch.tensor([
            1.0 if meta["node"] % 2 == 0 else -1.0 for meta in metadata
        ], dtype=torch.float64)
    raise KeyError(name)


def structured_filter_bank(
    active: list[int], device: str,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    if len(active) < 2:
        raise ValueError("structured filters require at least two active coordinates")
    metadata = [f06.coordinate_metadata(coord) for coord in active]
    bank: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    for name in FILTER_NAMES:
        raw = raw_values(name, metadata)
        centered = name in CENTERED_FILTERS
        values = raw - raw.mean() if centered else raw
        rms = values.square().mean().sqrt()
        if not torch.isfinite(rms) or float(rms) <= 0.0:
            raise RuntimeError(f"degenerate structured filter: {name}")
        values = values / rms
        full = torch.zeros((1, 31), dtype=torch.float32, device=device)
        full[0, active] = values.to(dtype=torch.float32, device=device)
        bank[name] = full
        for position, (meta, raw_value, value) in enumerate(
            zip(metadata, raw.tolist(), values.tolist())
        ):
            rows.append({
                "filter": name,
                **meta,
                "active_position": position,
                "normalization": "zero-mean-rms" if centered else "energy-rms",
                "raw_value": raw_value,
                "normalized_value": value,
                "centered": int(centered),
            })
    return bank, rows


def main() -> None:
    f06.CLAIM = CLAIM
    f06.FILTER_NAMES = FILTER_NAMES
    f06.CENTERED_FILTERS = CENTERED_FILTERS
    f06.INTERFERENCE_PAIRS = INTERFERENCE_PAIRS
    f06.INTERFERENCE_EPSILONS = (0.01, 0.1)
    f06.analytic_filter_bank = structured_filter_bank
    f06.main()


if __name__ == "__main__":
    main()
