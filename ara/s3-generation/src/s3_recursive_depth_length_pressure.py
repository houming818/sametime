#!/usr/bin/env python3
"""Frozen proof of a differentiable depth-conditioned EOS pressure board."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_recursive_depth_probability_exposure as d03  # noqa: E402
import s3_recursive_depth_free_generation_length as d05  # noqa: E402
import s3_multilevel_read_ablation_c12 as c12  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-RECURSIVE-DEPTH-LENGTH-PRESSURE-D06"
DEPTHS = (5, 6, 7)
FULL_DEPTH = 7


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def depth_budgets(source_lengths, depth: int, min_budget: int, max_budget: int):
    content = (source_lengths - 2).clamp_min(1).to(torch.float32)
    scale = 2.0 ** (depth - FULL_DEPTH)
    return torch.ceil(content * scale).to(torch.long).clamp(min=min_budget, max=max_budget)


def pressure_gate(step: int, budgets, onset: float, sharpness: float):
    progress = float(step + 1) / budgets.to(torch.float32)
    return torch.sigmoid(sharpness * (progress - onset))


def mix_eos_probability(probabilities, gate, eos: int):
    mixed = probabilities * (1.0 - gate[:, None])
    mixed[:, eos] = mixed[:, eos] + gate
    return mixed


def sample_top_p(probabilities, top_p: float, generator):
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
        dim=-1, keepdim=True,
    ).clamp_min(1e-12)
    selected = torch.multinomial(sorted_probabilities, 1, generator=generator)
    return sorted_indices.gather(1, selected).squeeze(1)


@torch.no_grad()
def generate_with_pressure(
    decoder, tree, masks, budgets, bos: int, eos: int, max_output: int, depth: int,
    onset: float, sharpness: float, temperature: float | None = None,
    top_p: float = 0.9, rng_seed: int = 0,
):
    batch = tree[0].shape[0]
    hidden = tree[0].new_zeros((batch, decoder.hidden))
    previous = torch.full((batch,), bos, dtype=torch.long, device=tree[0].device)
    done = torch.zeros(batch, dtype=torch.bool, device=tree[0].device)
    forced = torch.zeros(batch, dtype=torch.bool, device=tree[0].device)
    outputs = []
    gates = []
    generator = None
    if temperature is not None:
        generator = torch.Generator(device=tree[0].device)
        generator.manual_seed(rng_seed)
    for step in range(max_output):
        context, _ = d03.read_capped(decoder, hidden, tree, masks, depth)
        hidden = decoder.cell(
            torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
        )
        logits = decoder.output(torch.cat((hidden, context), dim=-1)).float()
        divisor = temperature if temperature is not None else 1.0
        probabilities = F.softmax(logits / divisor, dim=-1)
        gate = pressure_gate(step, budgets, onset, sharpness)
        mixed = mix_eos_probability(probabilities, gate, eos)
        if temperature is None:
            current = mixed.argmax(-1)
        else:
            current = sample_top_p(mixed, top_p, generator)
        wall = step >= budgets
        forced_now = wall & ~done
        current = torch.where(wall | done, torch.full_like(current, eos), current)
        outputs.append(current)
        gates.append(gate)
        forced = forced | forced_now
        done = done | current.eq(eos)
        previous = current
        if bool(done.all()):
            break
    return torch.stack(outputs, dim=1), forced, torch.stack(gates, dim=1)


def augment_metrics(base, lengths, budgets, forced):
    ratios = [length / max(1, budget) for length, budget in zip(lengths, budgets)]
    return {
        **base,
        "mean_budget": statistics.fmean(budgets) if budgets else 0.0,
        "median_budget": statistics.median(budgets) if budgets else 0.0,
        "mean_length_over_budget": statistics.fmean(ratios) if ratios else 0.0,
        "median_length_over_budget": statistics.median(ratios) if ratios else 0.0,
        "hard_wall_rate": sum(forced) / max(1, len(forced)),
        "pre_wall_eos_rate": 1.0 - sum(forced) / max(1, len(forced)),
    }


def paired_metrics(records, mode: str):
    ratios_57 = []
    ratios_67 = []
    monotonic = 0
    strict = 0
    total = 0
    for record in records:
        if mode == "greedy":
            triples = [[record["depths"][str(depth)]["greedy_length"] for depth in DEPTHS]]
        else:
            triples = list(zip(*[
                record["depths"][str(depth)]["sample_lengths"] for depth in DEPTHS
            ]))
        for length5, length6, length7 in triples:
            total += 1
            monotonic += int(length5 <= length6 <= length7)
            strict += int(length5 < length6 < length7)
            if length7 > 0:
                ratios_57.append(length5 / length7)
                ratios_67.append(length6 / length7)
    return {
        "pairs": total,
        "monotonic_fraction": monotonic / max(1, total),
        "strict_monotonic_fraction": strict / max(1, total),
        "median_l5_over_l7": statistics.median(ratios_57) if ratios_57 else None,
        "median_l6_over_l7": statistics.median(ratios_67) if ratios_67 else None,
    }


def baseline_for_seed(root: Path, seed: int):
    path = root / f"seed_{seed}" / "summary.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "test_row_sha256": row["test_row_sha256"],
        "depth7_sample_bleu4": row["metrics"]["7"]["sample"]["token_bleu4"],
        "depth7_sample_repetition": row["metrics"]["7"]["sample"]["adjacent_repetition_rate"],
    }


@torch.no_grad()
def probe_checkpoint(checkpoint_path, output, args, sp, pieces, pad, bos, eos, vocab):
    model, saved, config, state_hash, parent_hash = d03.load_model(
        checkpoint_path, args, sp, pad, vocab,
    )
    seed = int(saved["config"]["seed"])
    baseline = baseline_for_seed(Path(args.baseline_root), seed)
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    test_rows = test_rows[:args.eval_rows]
    row_hash = c12.rows_sha256(test_rows)
    depth_count = model.decoder.depth_embedding.num_embeddings
    if max(DEPTHS) >= depth_count:
        raise RuntimeError(f"requested depth {max(DEPTHS)} but checkpoint has {depth_count} levels")
    if baseline["test_row_sha256"] != row_hash:
        raise RuntimeError("D05 baseline and D06 test rows differ")

    cells = {
        depth: {
            "greedy": [], "greedy_refs": [], "greedy_budgets": [], "greedy_forced": [],
            "sample": [], "sample_refs": [], "sample_budgets": [], "sample_forced": [],
        }
        for depth in DEPTHS
    }
    records = []
    for batch_start in range(0, len(test_rows), args.batch_size):
        rows = test_rows[batch_start:batch_start + args.batch_size]
        source, lengths, target = c10.collate_rows(rows, pad, args.device)
        levels, masks = d03.condition_states(model, source, lengths, "native")
        tree = model.decoder.convolve(levels, masks)
        references = [c10.wmt.clean(row.tolist(), eos, pieces) for row in target.cpu()]
        batch_records = []
        for row_index, row in enumerate(rows):
            direction = row[2]
            batch_records.append({
                "row_index": batch_start + row_index,
                "direction": direction,
                "source": row[3][1] if direction == "en2zh" else row[3][0],
                "source_encoded_length": int(lengths[row_index]),
                "reference": sp.decode(references[row_index]),
                "reference_length": len(references[row_index]),
                "depths": {},
            })
        sample_tree, sample_masks = d05.repeat_tree(tree, masks, args.samples)
        for depth in DEPTHS:
            budgets = depth_budgets(lengths, depth, args.min_budget, args.max_output - 1)
            sample_budgets = budgets.repeat_interleave(args.samples)
            greedy_tensor, greedy_forced, _ = generate_with_pressure(
                model.decoder, tree, masks, budgets, bos, eos, args.max_output, depth,
                args.pressure_onset, args.pressure_sharpness,
            )
            sample_tensor, sample_forced, _ = generate_with_pressure(
                model.decoder, sample_tree, sample_masks, sample_budgets,
                bos, eos, args.max_output, depth, args.pressure_onset,
                args.pressure_sharpness, args.temperature, args.top_p,
                args.sample_seed + batch_start,
            )
            greedy_sequences = greedy_tensor.cpu().tolist()
            sample_sequences = sample_tensor.cpu().tolist()
            greedy_forced_rows = greedy_forced.cpu().tolist()
            sample_forced_rows = sample_forced.cpu().tolist()
            budget_rows = budgets.cpu().tolist()
            sample_budget_rows = sample_budgets.cpu().tolist()
            cells[depth]["greedy"].extend(greedy_sequences)
            cells[depth]["greedy_refs"].extend(references)
            cells[depth]["greedy_budgets"].extend(budget_rows)
            cells[depth]["greedy_forced"].extend(greedy_forced_rows)
            cells[depth]["sample"].extend(sample_sequences)
            cells[depth]["sample_refs"].extend([
                reference for reference in references for _ in range(args.samples)
            ])
            cells[depth]["sample_budgets"].extend(sample_budget_rows)
            cells[depth]["sample_forced"].extend(sample_forced_rows)
            for row_index, record in enumerate(batch_records):
                greedy_ids, _ = d05.clean(greedy_sequences[row_index], eos, pieces)
                start = row_index * args.samples
                stop = start + args.samples
                sample_ids = [d05.clean(value, eos, pieces)[0] for value in sample_sequences[start:stop]]
                record["depths"][str(depth)] = {
                    "budget": budget_rows[row_index],
                    "greedy_length": len(greedy_ids),
                    "greedy_hard_wall": greedy_forced_rows[row_index],
                    "greedy": sp.decode(greedy_ids),
                    "sample_lengths": [len(value) for value in sample_ids],
                    "sample_hard_wall": sample_forced_rows[start:stop],
                    "samples": [sp.decode(value) for value in sample_ids[:args.example_samples]],
                }
        records.extend(batch_records)

    metrics = {}
    for depth in DEPTHS:
        greedy, _, greedy_lengths, _ = d05.sequence_diagnostics(
            cells[depth]["greedy"], cells[depth]["greedy_refs"],
            eos, pieces, args.max_output,
        )
        sample, _, sample_lengths, _ = d05.sequence_diagnostics(
            cells[depth]["sample"], cells[depth]["sample_refs"],
            eos, pieces, args.max_output,
        )
        metrics[str(depth)] = {
            "greedy": augment_metrics(
                greedy, greedy_lengths, cells[depth]["greedy_budgets"],
                cells[depth]["greedy_forced"],
            ),
            "sample": augment_metrics(
                sample, sample_lengths, cells[depth]["sample_budgets"],
                cells[depth]["sample_forced"],
            ),
        }

    paired = {"greedy": paired_metrics(records, "greedy"), "sample": paired_metrics(records, "sample")}
    medians = [metrics[str(depth)]["sample"]["median_length"] for depth in DEPTHS]
    p1 = medians[0] < medians[1] < medians[2] and paired["sample"]["monotonic_fraction"] >= 0.80
    ratio57 = paired["sample"]["median_l5_over_l7"]
    ratio67 = paired["sample"]["median_l6_over_l7"]
    p2 = (
        ratio57 is not None and ratio67 is not None
        and 0.15 <= ratio57 <= 0.40 and 0.35 <= ratio67 <= 0.70
    )
    p3 = all(
        metrics[str(depth)]["sample"]["pre_wall_eos_rate"] >= 0.65
        and metrics[str(depth)]["sample"]["nonempty_rate"] >= 0.95
        for depth in DEPTHS
    )
    depth7 = metrics["7"]["sample"]
    p4 = (
        depth7["token_bleu4"] >= baseline["depth7_sample_bleu4"] - 1.0
        and depth7["adjacent_repetition_rate"] <= baseline["depth7_sample_repetition"] + 0.05
    )
    p0 = (
        len(test_rows) == args.eval_rows and baseline["test_row_sha256"] == row_hash
        and all(math.isfinite(value) for value in medians)
        and all(
            all(record["depths"][str(left)]["budget"] <= record["depths"][str(right)]["budget"]
                for left, right in ((5, 6), (6, 7)))
            for record in records
        )
    )
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_sha256": state_hash,
        "parent_state_sha256": parent_hash,
        "seed": seed,
        "test_rows": len(test_rows),
        "test_row_sha256": row_hash,
        "depth_count": depth_count,
        "depths": list(DEPTHS),
        "pressure_contract": {
            "budget_from_target": False,
            "source_special_tokens_removed": 2,
            "min_budget": args.min_budget,
            "max_output": args.max_output,
            "onset": args.pressure_onset,
            "sharpness": args.pressure_sharpness,
            "probability_mixture": "P'=(1-g)P+g*delta_EOS",
            "hard_wall": True,
            "samples_per_row": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "target_read_during_generation": False,
            "teacher_forcing": False,
        },
        "baseline": baseline,
        "metrics": metrics,
        "paired": paired,
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4},
        "examples": records[:args.example_rows],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "records.json", records)
    return summary


def aggregate(summaries, output: Path, args):
    seeds = [row["seed"] for row in summaries]
    row_hashes = {row["test_row_sha256"] for row in summaries}
    pass_count = {
        gate: sum(row["gates"][gate] for row in summaries)
        for gate in ("P0", "P1", "P2", "P3", "P4")
    }
    reverse = 0
    for row in summaries:
        medians = [row["metrics"][str(depth)]["sample"]["median_length"] for depth in DEPTHS]
        reverse += int(medians[0] > medians[1] > medians[2])
    p0 = len(row_hashes) == 1 and pass_count["P0"] == len(summaries)
    p1 = pass_count["P1"] >= min(2, len(summaries))
    p2 = pass_count["P2"] >= min(2, len(summaries))
    p3 = pass_count["P3"] >= min(2, len(summaries))
    p4 = pass_count["P4"] >= min(2, len(summaries))
    p5 = p1 and p2 and p3 and reverse == 0
    if not p0:
        decision = "invalid_contract"
    elif p1 and p2 and p3 and p4 and p5:
        decision = "supported_depth_length_pressure_board"
    elif p1 and p2:
        decision = "partial_hard_capacity_coupling_only"
    else:
        decision = "not_supported_depth_length_pressure_board"
    result = {
        "claim": CLAIM,
        "mode": args.mode,
        "seeds": seeds,
        "test_row_sha256": next(iter(row_hashes)) if len(row_hashes) == 1 else sorted(row_hashes),
        "pressure_contract": summaries[0]["pressure_contract"],
        "seed_results": [{
            "seed": row["seed"],
            "gates": row["gates"],
            "sample_median_lengths": {
                str(depth): row["metrics"][str(depth)]["sample"]["median_length"]
                for depth in DEPTHS
            },
            "sample_median_budgets": {
                str(depth): row["metrics"][str(depth)]["sample"]["median_budget"]
                for depth in DEPTHS
            },
            "sample_pre_wall_eos": {
                str(depth): row["metrics"][str(depth)]["sample"]["pre_wall_eos_rate"]
                for depth in DEPTHS
            },
            "paired_sample": row["paired"]["sample"],
            "depth7_safety": {
                "baseline_bleu4": row["baseline"]["depth7_sample_bleu4"],
                "pressure_bleu4": row["metrics"]["7"]["sample"]["token_bleu4"],
                "baseline_repetition": row["baseline"]["depth7_sample_repetition"],
                "pressure_repetition": row["metrics"]["7"]["sample"]["adjacent_repetition_rate"],
            },
        } for row in summaries],
        "gate_pass_counts": pass_count,
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5},
        "decision": decision,
    }
    write_json(output / "summary.json", result)
    return result


def self_test(output: Path):
    probabilities = torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float64)
    gate = torch.tensor([0.25], dtype=torch.float64)
    mixed = mix_eos_probability(probabilities, gate, eos=0)
    assert torch.allclose(mixed.sum(-1), torch.ones(1, dtype=torch.float64))
    assert abs(float(mixed[0, 1] / mixed[0, 2]) - 2.0) < 1e-12
    budgets = depth_budgets(torch.tensor([18]), 5, 2, 127)
    assert int(budgets[0]) == 4
    gates = [float(pressure_gate(step, budgets, 0.85, 20.0)[0]) for step in range(4)]
    assert gates == sorted(gates)
    write_json(output / "self_test.json", {
        "claim": CLAIM,
        "passed": True,
        "mixed": mixed.tolist(),
        "non_eos_ratio_preserved": True,
        "depth5_budget_for_encoded_length18": int(budgets[0]),
        "pressure_gates": gates,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", default=[])
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--baseline-root", default="ara/s3-generation/evidence/s3_recursive_depth_free_generation_length_d05/formal")
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--max-output", type=int, default=128)
    parser.add_argument("--min-budget", type=int, default=2)
    parser.add_argument("--pressure-onset", type=float, default=0.85)
    parser.add_argument("--pressure-sharpness", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-seed", type=int, default=17601)
    parser.add_argument("--example-rows", type=int, default=8)
    parser.add_argument("--example-samples", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        self_test(output)
        return
    if not args.checkpoints:
        parser.error("at least one --checkpoint is required unless --self-test is used")
    if args.mode == "smoke":
        args.eval_rows = args.eval_rows or 16
        args.batch_size = args.batch_size or 4
        args.samples = args.samples or 3
    else:
        args.eval_rows = args.eval_rows or 128
        args.batch_size = args.batch_size or 8
        args.samples = args.samples or 8
    random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    torch.cuda.manual_seed_all(args.sample_seed)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    vocab = pieces + 3
    summaries = []
    for checkpoint_name in args.checkpoints:
        checkpoint = Path(checkpoint_name)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seed = int(saved["config"]["seed"])
        summary = probe_checkpoint(
            checkpoint, output / f"seed_{seed}", args, sp, pieces, pad, bos, eos, vocab,
        )
        summaries.append(summary)
        print(json.dumps({
            "event": "checkpoint_complete",
            "seed": seed,
            "medians": {
                str(depth): summary["metrics"][str(depth)]["sample"]["median_length"]
                for depth in DEPTHS
            },
            "pre_wall_eos": {
                str(depth): summary["metrics"][str(depth)]["sample"]["pre_wall_eos_rate"]
                for depth in DEPTHS
            },
            "gates": summary["gates"],
        }, ensure_ascii=False), flush=True)
        del saved
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = aggregate(summaries, output, args)
    print(json.dumps({"event": "D06_complete", **result["gates"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
