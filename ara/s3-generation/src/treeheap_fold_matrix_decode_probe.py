#!/usr/bin/env python3
"""Decode a frozen TreeHeap checkpoint under scaled FOLD observation matrices."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch

import s3_recursive_depth_pressure_protocol_training as d07
import s3_structural_protocol_full_pipeline_d10 as d10
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


DEFAULT_SCALES = (0.0, 0.25, 0.5, math.sqrt(0.5), 1.0)


def scaled_fold(slots: torch.Tensor, slot_mask: torch.Tensor, scale: float):
    """Build the production TreeHeap with [s I, s I] at every two-child merge."""
    levels = [slots]
    masks = [slot_mask]
    node, valid = slots, slot_mask
    while node.shape[1] > 1:
        left, right = node[:, 0::2], node[:, 1::2]
        left_valid, right_valid = valid[:, 0::2], valid[:, 1::2]
        both = left_valid & right_valid
        parent = torch.where(
            both[:, :, None],
            (left + right) * scale,
            torch.where(left_valid[:, :, None], left, right),
        )
        valid = left_valid | right_valid
        parent = parent * valid[:, :, None]
        levels.append(parent)
        masks.append(valid)
        node = parent
    return list(reversed(levels)), list(reversed(masks))


@contextmanager
def fold_scale(scale: float):
    original = d07.fold_protocol
    d07.fold_protocol = lambda slots, mask: scaled_fold(slots, mask, scale)
    try:
        yield
    finally:
        d07.fold_protocol = original


@contextmanager
def decoder_up_mode(model, mode: str):
    """Keep or bypass learned decoder-side K_up without changing its weights."""
    modules = [model.reconstructor]
    if model.extra is not None:
        modules.append(model.extra.reconstructor)
    previous = [module.use_up for module in modules]
    if mode == "bypass":
        for module in modules:
            module.use_up = False
    elif mode != "native":
        raise ValueError(mode)
    try:
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.use_up = enabled


def adjacent_repetition(ids: list[int]) -> float:
    if len(ids) < 2:
        return 0.0
    return sum(left == right for left, right in zip(ids, ids[1:])) / (len(ids) - 1)


def character_f2(candidate: str, reference: str, max_order: int = 6) -> float:
    """Dependency-free character n-gram F2 diagnostic; this is not sacreBLEU chrF."""
    candidate = "".join(candidate.split())
    reference = "".join(reference.split())
    scores = []
    for order in range(1, max_order + 1):
        cand = Counter(candidate[i:i + order] for i in range(max(0, len(candidate) - order + 1)))
        ref = Counter(reference[i:i + order] for i in range(max(0, len(reference) - order + 1)))
        overlap = sum((cand & ref).values())
        precision = overlap / max(1, sum(cand.values()))
        recall = overlap / max(1, sum(ref.values()))
        scores.append(5 * precision * recall / max(1e-12, 4 * precision + recall))
    return sum(scores) / len(scores)


@torch.inference_mode()
def decode(specimen: dict, depth: int, scale: float, up_mode: str, runtime, max_len: int):
    payload, args, sp, model, pieces, eos, bos, _ = runtime
    text = specimen["source"]
    direction = specimen["direction"]
    raw = sp.encode(text.strip(), out_type=int)
    direction_id = {"en2zh": pieces + 1, "zh2en": pieces + 2}[direction]
    source_ids = [direction_id, *raw[:32], eos]
    source = torch.tensor([source_ids], dtype=torch.long, device=args.device)
    lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=args.device)
    with fold_scale(scale), decoder_up_mode(model, up_mode):
        generated, route, budgets, _, _ = model.greedy(
            source, lengths, bos, eos, max_len, depth,
        )
    raw_output = generated[0].tolist()
    eos_reached = eos in raw_output
    clean = d10.wmt.clean(raw_output, eos, pieces)
    output = sp.decode(clean)
    reference = specimen.get("reference", "")
    reference_pieces = len(sp.encode(reference, out_type=int)) if reference else 0
    length_ratio = len(clean) / max(1, reference_pieces) if reference else None
    repetition = adjacent_repetition(clean)
    formed = bool(clean) and eos_reached and repetition <= 0.2
    if length_ratio is not None:
        formed = formed and 0.2 <= length_ratio <= 2.5
    return {
        "arm": payload["arm"],
        "step": int(payload["step"]),
        "direction": direction,
        "depth": depth,
        "up_mode": up_mode,
        "fold_scale": scale,
        "is_native_scale": math.isclose(scale, math.sqrt(0.5), rel_tol=0.0, abs_tol=1e-12),
        "specimen_id": specimen.get("id", "manual"),
        "category": specimen.get("category", "manual"),
        "input": text,
        "reference": reference,
        "input_pieces": len(raw),
        "input_truncated": len(raw) > 32,
        "output": output,
        "output_pieces": len(clean),
        "reference_pieces": reference_pieces,
        "output_reference_length_ratio": length_ratio,
        "eos_reached": eos_reached,
        "unique_piece_ratio": len(set(clean)) / max(1, len(clean)),
        "adjacent_repetition_rate": repetition,
        "character_f2_diagnostic": character_f2(output, reference) if reference else None,
        "formed": formed,
        "budget": int(budgets[0]),
        "route_depth_mass": [float(value) for value in route.detach().cpu()],
    }


def load_specimens(path: str, manual_cases: list[list[str]] | None) -> list[dict]:
    specimens = []
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("direction") not in ("en2zh", "zh2en") or not row.get("source"):
                    raise ValueError(f"invalid specimen at line {line_number}")
                specimens.append(row)
    for direction, source in manual_cases or []:
        specimens.append({"direction": direction, "source": source})
    if not specimens:
        raise ValueError("at least one --specimens file or --case is required")
    return specimens


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["up_mode"], row["depth"], row["fold_scale"])].append(row)
    summary = []
    for (up_mode, depth, scale), group in sorted(groups.items()):
        diagnostics = [row["character_f2_diagnostic"] for row in group if row["character_f2_diagnostic"] is not None]
        summary.append({
            "up_mode": up_mode,
            "depth": depth,
            "fold_scale": scale,
            "specimens": len(group),
            "formed_rate": sum(row["formed"] for row in group) / len(group),
            "empty_rate": sum(row["output_pieces"] == 0 for row in group) / len(group),
            "eos_rate": sum(row["eos_reached"] for row in group) / len(group),
            "mean_output_pieces": sum(row["output_pieces"] for row in group) / len(group),
            "mean_character_f2_diagnostic": sum(diagnostics) / len(diagnostics) if diagnostics else None,
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--warm-start", default="")
    parser.add_argument("--spm-model", default="")
    parser.add_argument("--case", action="append", nargs=2, metavar=("DIRECTION", "TEXT"))
    parser.add_argument("--specimens", default="")
    parser.add_argument("--scale", action="append", type=float)
    parser.add_argument("--up-mode", action="append", choices=("native", "bypass"))
    parser.add_argument("--depth", choices=("all", "5", "6", "7"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit-specimens", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    for direction, _ in args.case or []:
        if direction not in ("en2zh", "zh2en"):
            parser.error(f"invalid direction: {direction}")

    specimens = load_specimens(args.specimens, args.case)
    if args.limit_specimens > 0:
        specimens = specimens[:args.limit_specimens]

    runtime = load_runtime(
        Path(args.checkpoint), args.device, args.source_checkpoint,
        args.warm_start, args.spm_model,
    )
    scales = tuple(args.scale) if args.scale else DEFAULT_SCALES
    up_modes = tuple(args.up_mode) if args.up_mode else ("native",)
    depths = DEPTHS if args.depth == "all" else (int(args.depth),)
    rows = [
        decode(specimen, depth, scale, up_mode, runtime, args.max_new_tokens)
        for specimen in specimens
        for depth in depths
        for scale in scales
        for up_mode in up_modes
    ]
    report = {
        "claim": "FOLD-OBSERVATION-MATRIX-DECODE-PROBE",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_sha256": runtime[-1],
        "matrix": "parent = [s I, s I] @ [left; right]",
        "native_scale": math.sqrt(0.5),
        "scales": list(scales),
        "up_modes": list(up_modes),
        "specimens": specimens,
        "primary_evidence": "decoded_text",
        "summary": summarize(rows),
        "results": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
