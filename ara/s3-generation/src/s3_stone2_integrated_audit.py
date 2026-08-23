#!/usr/bin/env python3
"""Frozen structural and reload audit for the STONE-2 integrated smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_hstate_multilevel_convolution as c11  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402
import s3_stone2_integrated_pipeline as integrated  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def group_norm(parameters) -> float:
    values = [
        parameter.grad.detach().square().sum()
        for parameter in parameters if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def gradient_probe(model, batch, pad, bos, device):
    model.train()
    source, length, target = c10.collate_rows(batch, pad, device)
    _, _, _, levels, masks = model.states(source, length)
    for level in levels:
        level.retain_grad()
    logits, route = model.decoder.teacher(levels, masks, target, bos)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=pad,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    level_norms = [float(level.grad.norm()) if level.grad is not None else 0.0 for level in levels]
    inner = model.encoder.inner
    groups = {
        "write_embedding": [inner.embedding.weight],
        "butterfly_forward": list(inner.communication.forward_kernel.parameters()),
        "butterfly_backward": list(inner.communication.backward_kernel.parameters()),
        "read_kernel": list(model.decoder.read_kernel.parameters()),
        "branch": list(model.decoder.branch.parameters()),
        "decoder_cell": list(model.decoder.cell.parameters()),
        "decoder_output": list(model.decoder.output.parameters()),
    }
    norms = {name: group_norm(parameters) for name, parameters in groups.items()}
    return {
        "loss": float(loss.detach()),
        "level_widths": [level.shape[1] for level in levels],
        "level_grad_norms": level_norms,
        "parameter_grad_norms": norms,
        "route": [float(value) for value in route.detach().cpu()],
        "finite": math.isfinite(float(loss.detach()))
        and all(math.isfinite(value) for value in [*level_norms, *norms.values()]),
    }


@torch.no_grad()
def closure_probe(model, rows, pad, device):
    source, length, _ = c10.collate_rows(rows, pad, device)
    leaf, root, details, scales, masks = model.encoder.fold(source, length)
    levels, _ = model.encoder.unfold(root, details, scales, masks)
    restored = levels[-1]
    active = masks[0][:, :, None]
    difference = (restored - leaf) * active
    return {
        "mse": float(difference.square().sum() / active.sum().clamp_min(1)),
        "max_abs": float(difference.abs().max()),
        "root_rms": float(root.square().mean().sqrt()),
    }


@torch.no_grad()
def signatures(model, rows, sp, pad, bos, eos, pieces, device, limit=16):
    result = []
    for row in rows[:limit]:
        source, length, target = c10.collate_rows([row], pad, device)
        generated, route = model.greedy(source, length, bos, eos, min(64, target.shape[1] + 16))
        ids = c10.wmt.clean(generated[0].tolist(), eos, pieces)
        result.append({
            "source_ids": row[0],
            "tokens": ids,
            "text": sp.decode(ids),
            "route": [float(value) for value in route.detach().cpu()],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-rows", type=int, default=256)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    config.device = args.device
    config.task_eval_rows = args.eval_rows
    sp = spm.SentencePieceProcessor(model_file=config.spm_model)
    pieces = sp.get_piece_size()
    pad, bos, eos = pieces, sp.bos_id(), sp.eos_id()
    vocab = pieces + 3
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}

    model = integrated.build_integrated_model(config, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(args.device)
    _, valid_rows, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    rows = test_rows[:args.eval_rows]

    native = c11.evaluate(model, rows, pad, bos, args.device, config.eval_batch)
    leaf = c11.evaluate(model, rows, pad, bos, args.device, config.eval_batch, mode="leaf_only")
    identity = c11.evaluate(
        model, rows, pad, bos, args.device, config.eval_batch, runtime_mode="identity",
    )
    source_shuffle = c11.evaluate(
        model, rows, pad, bos, args.device, config.eval_batch, intervention="source_shuffle",
    )
    pair_break = c11.evaluate(
        model, rows, pad, bos, args.device, config.eval_batch, pair_break_depth=0,
    )
    depth_rows = []
    for depth in range(int(math.log2(config.heap_width)) + 1):
        measured = c11.evaluate(
            model, rows, pad, bos, args.device, config.eval_batch, ablate_depth=depth,
        )
        depth_rows.append({"depth": depth, "nll": measured["nll"], "damage": measured["nll"] - native["nll"]})

    first_signatures = signatures(model, rows, sp, pad, bos, eos, pieces, args.device)
    reloaded = integrated.build_integrated_model(config, vocab, pad)
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    reloaded.to(args.device)
    second_signatures = signatures(reloaded, rows, sp, pad, bos, eos, pieces, args.device)
    unique_outputs = len({tuple(row["tokens"]) for row in first_signatures})
    nonempty = sum(bool(row["tokens"]) for row in first_signatures) / max(1, len(first_signatures))
    severe = sum(
        sum(a == b for a, b in zip(row["tokens"], row["tokens"][1:]))
        / max(1, len(row["tokens"]) - 1) >= 0.5
        for row in first_signatures
    ) / max(1, len(first_signatures))

    gradient = gradient_probe(model, valid_rows[:config.task_batch], pad, bos, args.device)
    closure = closure_probe(model, rows[:config.task_batch], pad, args.device)
    positive_parent_depths = sum(
        row["damage"] > 0.0 for row in depth_rows[:-1]
    )
    no_stop = not any("stop" in name.lower() for name, _ in model.named_parameters())
    gates = {
        "S1_closure": closure["max_abs"] < 1e-4,
        "S2_no_stop": no_stop,
        "S3_all_levels_receive_gradient": gradient["finite"] and all(
            value > 0.0 for value in gradient["level_grad_norms"]
        ),
        "S4_trainable_groups_receive_gradient": all(
            value > 0.0 for value in gradient["parameter_grad_norms"].values()
        ),
        "S6_structural_interventions_positive": all(
            value > 0.0 for value in (
                identity["nll"] - native["nll"],
                source_shuffle["nll"] - native["nll"],
                pair_break["nll"] - native["nll"],
            )
        ),
        "S7_multilevel_positive": leaf["nll"] > native["nll"] and positive_parent_depths >= 2,
        "S8_generation_not_fixed": nonempty == 1.0 and unique_outputs >= 2 and severe < 1.0,
        "S9_reload_exact": first_signatures == second_signatures,
    }
    result = {
        "claim": integrated.CLAIM,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rows": len(rows),
        "native": native,
        "interventions": {
            "leaf_only_damage": leaf["nll"] - native["nll"],
            "identity_damage": identity["nll"] - native["nll"],
            "source_shuffle_damage": source_shuffle["nll"] - native["nll"],
            "pair_break_damage": pair_break["nll"] - native["nll"],
            "depths": depth_rows,
        },
        "closure": closure,
        "gradient": gradient,
        "generation": {
            "nonempty": nonempty,
            "severe_repetition": severe,
            "unique_outputs": unique_outputs,
            "examples": first_signatures,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    write_json(args.output, result)
    print(json.dumps({"event": "integrated_audit", "gates": gates}, ensure_ascii=False), flush=True)
    if not result["all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
