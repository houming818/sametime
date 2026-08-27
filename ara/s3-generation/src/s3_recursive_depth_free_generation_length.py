#!/usr/bin/env python3
"""Free-generation length response under frozen cumulative TreeHeap READ depths."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
import s3_multilevel_read_ablation_c12 as c12  # noqa: E402
import s3_pretrain_task_posterior_pipeline as c10  # noqa: E402


CLAIM = "S3-RECURSIVE-DEPTH-FREE-LENGTH-D05"
DEPTHS = (5, 6, 7)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_ratio(numerator: int, denominator: int):
    return float(numerator) / denominator if denominator > 0 else None


def clean(ids, eos: int, pieces: int):
    output = []
    eos_hit = False
    for token in ids:
        token = int(token)
        if token == eos:
            eos_hit = True
            break
        if 0 <= token < pieces:
            output.append(token)
    return output, eos_hit


def repeat_tree(tree, masks, repeats: int):
    return (
        [level.repeat_interleave(repeats, dim=0) for level in tree],
        [mask.repeat_interleave(repeats, dim=0) for mask in masks],
    )


def sample_top_p(logits, temperature: float, top_p: float, generator):
    probabilities = F.softmax(logits / temperature, dim=-1)
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
def generate_capped(
    decoder, tree, masks, bos: int, eos: int, max_output: int, depth: int,
    temperature: float | None = None, top_p: float | None = None, rng_seed: int = 0,
):
    batch = tree[0].shape[0]
    hidden = tree[0].new_zeros((batch, decoder.hidden))
    previous = torch.full((batch,), bos, dtype=torch.long, device=tree[0].device)
    done = torch.zeros(batch, dtype=torch.bool, device=tree[0].device)
    outputs = []
    generator = None
    if temperature is not None:
        generator = torch.Generator(device=tree[0].device)
        generator.manual_seed(rng_seed)
    for _ in range(max_output):
        context, _ = d03.read_capped(decoder, hidden, tree, masks, depth)
        hidden = decoder.cell(
            torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
        )
        logits = decoder.output(torch.cat((hidden, context), dim=-1)).float()
        if temperature is None:
            current = logits.argmax(-1)
        else:
            current = sample_top_p(logits, temperature, top_p, generator)
        current = torch.where(done, torch.full_like(current, eos), current)
        outputs.append(current)
        done = done | current.eq(eos)
        previous = current
        if bool(done.all()):
            break
    return torch.stack(outputs, dim=1)


def sequence_diagnostics(sequences, references, eos: int, pieces: int, max_output: int):
    tokens = []
    lengths = []
    eos_hits = []
    adjacent_equal = 0
    adjacent_total = 0
    max_token_shares = []
    unique_token_fractions = []
    for sequence in sequences:
        row, eos_hit = clean(sequence, eos, pieces)
        tokens.append(row)
        lengths.append(len(row))
        eos_hits.append(eos_hit)
        adjacent_equal += sum(left == right for left, right in zip(row, row[1:]))
        adjacent_total += max(0, len(row) - 1)
        counts = Counter(row)
        max_token_shares.append(max(counts.values()) / len(row) if row else 0.0)
        unique_token_fractions.append(len(counts) / len(row) if row else 0.0)
    return {
        "count": len(lengths),
        "mean_length": statistics.fmean(lengths) if lengths else 0.0,
        "median_length": statistics.median(lengths) if lengths else 0.0,
        "stdev_length": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "p25_length": percentile(lengths, 0.25),
        "p75_length": percentile(lengths, 0.75),
        "min_length": min(lengths, default=0),
        "max_length": max(lengths, default=0),
        "eos_hit_rate": sum(eos_hits) / max(1, len(eos_hits)),
        "cap_rate": sum(not value for value in eos_hits) / max(1, len(eos_hits)),
        "nonempty_rate": sum(value > 0 for value in lengths) / max(1, len(lengths)),
        "adjacent_repetition_rate": adjacent_equal / max(1, adjacent_total),
        "mean_max_token_share": statistics.fmean(max_token_shares) if max_token_shares else 0.0,
        "mean_unique_token_fraction": (
            statistics.fmean(unique_token_fractions) if unique_token_fractions else 0.0
        ),
        "token_bleu4": c10.wmt_metrics.bleu4(tokens, references),
        "max_output": max_output,
    }, tokens, lengths, eos_hits


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
            ratio57 = safe_ratio(length5, length7)
            ratio67 = safe_ratio(length6, length7)
            if ratio57 is not None:
                ratios_57.append(ratio57)
            if ratio67 is not None:
                ratios_67.append(ratio67)
    return {
        "pairs": total,
        "monotonic_fraction": monotonic / max(1, total),
        "strict_monotonic_fraction": strict / max(1, total),
        "median_l5_over_l7": statistics.median(ratios_57) if ratios_57 else None,
        "median_l6_over_l7": statistics.median(ratios_67) if ratios_67 else None,
        "valid_l5_over_l7_pairs": len(ratios_57),
        "valid_l6_over_l7_pairs": len(ratios_67),
    }


@torch.no_grad()
def probe_checkpoint(checkpoint_path, output, args, sp, pieces, pad, bos, eos, vocab):
    model, saved, config, state_hash, parent_hash = d03.load_model(
        checkpoint_path, args, sp, pad, vocab,
    )
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    _, _, test_rows = c10.collect_wmt_rows(config, sp, direction_ids, eos)
    test_rows = test_rows[:args.eval_rows]
    row_hash = c12.rows_sha256(test_rows)
    depth_count = model.decoder.depth_embedding.num_embeddings
    if max(DEPTHS) >= depth_count:
        raise RuntimeError(f"requested depth {max(DEPTHS)} but checkpoint has {depth_count} levels")

    cells = {
        depth: {"greedy": [], "greedy_refs": [], "sample": [], "sample_refs": []}
        for depth in DEPTHS
    }
    records = []
    for batch_start in range(0, len(test_rows), args.batch_size):
        rows = test_rows[batch_start:batch_start + args.batch_size]
        source, length, target = c10.collate_rows(rows, pad, args.device)
        levels, masks = d03.condition_states(model, source, length, "native")
        tree = model.decoder.convolve(levels, masks)
        references = [c10.wmt.clean(row.tolist(), eos, pieces) for row in target.cpu()]
        batch_records = []
        for row_index, row in enumerate(rows):
            direction = row[2]
            batch_records.append({
                "row_index": batch_start + row_index,
                "direction": direction,
                "source": row[3][1] if direction == "en2zh" else row[3][0],
                "reference": sp.decode(references[row_index]),
                "reference_length": len(references[row_index]),
                "depths": {},
            })

        sample_tree, sample_masks = repeat_tree(tree, masks, args.samples)
        for depth in DEPTHS:
            greedy_tensor = generate_capped(
                model.decoder, tree, masks, bos, eos, args.max_output, depth,
            ).cpu()
            sample_tensor = generate_capped(
                model.decoder, sample_tree, sample_masks, bos, eos, args.max_output, depth,
                args.temperature, args.top_p, args.sample_seed + batch_start,
            ).cpu()
            greedy_sequences = greedy_tensor.tolist()
            sample_sequences = sample_tensor.tolist()
            repeated_references = [reference for reference in references for _ in range(args.samples)]
            cells[depth]["greedy"].extend(greedy_sequences)
            cells[depth]["greedy_refs"].extend(references)
            cells[depth]["sample"].extend(sample_sequences)
            cells[depth]["sample_refs"].extend(repeated_references)
            for row_index, record in enumerate(batch_records):
                greedy_ids, greedy_eos = clean(greedy_sequences[row_index], eos, pieces)
                start = row_index * args.samples
                stop = start + args.samples
                sample_rows = [clean(value, eos, pieces) for value in sample_sequences[start:stop]]
                record["depths"][str(depth)] = {
                    "greedy_length": len(greedy_ids),
                    "greedy_eos": greedy_eos,
                    "greedy": sp.decode(greedy_ids),
                    "sample_lengths": [len(value[0]) for value in sample_rows],
                    "sample_eos": [value[1] for value in sample_rows],
                    "samples": [sp.decode(value[0]) for value in sample_rows[:args.example_samples]],
                }
        records.extend(batch_records)

    metrics = {}
    for depth in DEPTHS:
        greedy, _, _, _ = sequence_diagnostics(
            cells[depth]["greedy"], cells[depth]["greedy_refs"], eos, pieces, args.max_output,
        )
        sample, _, _, _ = sequence_diagnostics(
            cells[depth]["sample"], cells[depth]["sample_refs"], eos, pieces, args.max_output,
        )
        metrics[str(depth)] = {"greedy": greedy, "sample": sample}

    paired = {
        "greedy": paired_metrics(records, "greedy"),
        "sample": paired_metrics(records, "sample"),
    }
    sample_medians = [metrics[str(depth)]["sample"]["median_length"] for depth in DEPTHS]
    p1 = (
        sample_medians[0] < sample_medians[1] < sample_medians[2]
        and paired["sample"]["monotonic_fraction"] >= 0.60
    )
    ratio57 = paired["sample"]["median_l5_over_l7"]
    ratio67 = paired["sample"]["median_l6_over_l7"]
    p2 = (
        ratio57 is not None and ratio67 is not None
        and 0.125 <= ratio57 <= 0.375
        and 0.35 <= ratio67 <= 0.65
    )
    p3 = all(metrics[str(depth)]["sample"]["cap_rate"] <= 0.10 for depth in DEPTHS)
    p0 = (
        len(test_rows) == args.eval_rows
        and all(math.isfinite(value) for value in sample_medians)
        and len({metric["sample"]["max_output"] for metric in metrics.values()}) == 1
    )
    summary = {
        "claim": CLAIM,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_state_sha256": state_hash,
        "parent_state_sha256": parent_hash,
        "seed": int(saved["config"]["seed"]),
        "test_rows": len(test_rows),
        "test_row_sha256": row_hash,
        "depth_count": depth_count,
        "depths": list(DEPTHS),
        "generation_contract": {
            "max_output": args.max_output,
            "samples_per_row": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "target_read_during_generation": False,
            "teacher_forcing": False,
            "target_grouping": False,
        },
        "metrics": metrics,
        "paired": paired,
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3},
        "examples": records[:args.example_rows],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "records.json", records)
    return summary


def aggregate(summaries, output: Path, args):
    row_hashes = {row["test_row_sha256"] for row in summaries}
    seeds = [row["seed"] for row in summaries]
    p1_count = sum(row["gates"]["P1"] for row in summaries)
    reverse_count = 0
    for row in summaries:
        medians = [row["metrics"][str(depth)]["sample"]["median_length"] for depth in DEPTHS]
        reverse_count += int(medians[0] > medians[1] > medians[2])
    p0 = len(row_hashes) == 1 and all(row["gates"]["P0"] for row in summaries)
    p1 = p1_count >= min(2, len(summaries))
    p2 = sum(row["gates"]["P2"] for row in summaries) >= min(2, len(summaries))
    p3 = all(row["gates"]["P3"] for row in summaries)
    p4 = p1 and reverse_count == 0
    if not p0:
        decision = "invalid_contract"
    elif not p3:
        decision = "inconclusive_length_censored"
    elif p1 and p2 and p4:
        decision = "supported_near_binary_length_scale"
    elif p1 and p4:
        decision = "supported_monotonic_length_response_only"
    else:
        decision = "not_supported_natural_length_response"
    result = {
        "claim": CLAIM,
        "mode": args.mode,
        "seeds": seeds,
        "test_row_sha256": next(iter(row_hashes)) if len(row_hashes) == 1 else sorted(row_hashes),
        "generation_contract": summaries[0]["generation_contract"],
        "seed_results": [{
            "seed": row["seed"],
            "gates": row["gates"],
            "sample_median_lengths": {
                str(depth): row["metrics"][str(depth)]["sample"]["median_length"]
                for depth in DEPTHS
            },
            "sample_cap_rates": {
                str(depth): row["metrics"][str(depth)]["sample"]["cap_rate"]
                for depth in DEPTHS
            },
            "paired_sample": row["paired"]["sample"],
            "paired_greedy": row["paired"]["greedy"],
        } for row in summaries],
        "gates": {"P0": p0, "P1": p1, "P2": p2, "P3": p3, "P4": p4},
        "decision": decision,
    }
    write_json(output / "summary.json", result)
    return result


def self_test(output: Path):
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    fake = [{"depths": {
        "5": {"greedy_length": 2, "sample_lengths": [2, 3]},
        "6": {"greedy_length": 4, "sample_lengths": [4, 5]},
        "7": {"greedy_length": 8, "sample_lengths": [8, 10]},
    }}]
    metric = paired_metrics(fake, "sample")
    assert metric["monotonic_fraction"] == 1.0
    assert metric["median_l5_over_l7"] == 0.275
    write_json(output / "self_test.json", {"claim": CLAIM, "passed": True, "paired": metric})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", default=[])
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--wmt-data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--eval-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--max-output", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-seed", type=int, default=17501)
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
    if not (0.0 < args.top_p <= 1.0 and args.temperature > 0.0):
        parser.error("temperature must be positive and top-p must be in (0, 1]")
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
            "cap_rates": {
                str(depth): summary["metrics"][str(depth)]["sample"]["cap_rate"]
                for depth in DEPTHS
            },
            "gates": summary["gates"],
        }, ensure_ascii=False), flush=True)
        del saved
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = aggregate(summaries, output, args)
    print(json.dumps({"event": "D05_complete", **result["gates"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
