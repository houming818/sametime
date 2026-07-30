#!/usr/bin/env python3
"""Observe one frozen C04 TreeHeap state through matched C06 decoders."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_adaptive_lifting_wmt as adaptive
import s3_private_protocol_data_dose as data_dose
import s3_stone1_decoder_depth_floor as c06
import s3_stone1_frozen_encoder_pressure_decoder as c05
import s3_stone1_private_protocol as c01
import s3_wmt_treeheap_seq2seq as base


ARMS = {
    "native_control": "native",
    "leaf_reference": "force_leaf",
    "depth_floor": "depth_floor",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    parser.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    parser.add_argument("--c04-checkpoint", required=True)
    parser.add_argument("--decoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--generation-samples", type=int, default=96)
    parser.add_argument("--generation-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--source-col", type=int, default=1)
    parser.add_argument("--target-col", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71900)
    parser.add_argument("--max-scan", type=int, default=300_000)
    parser.add_argument("--heap-width", type=int, default=64)
    parser.add_argument("--leaf-cut", type=int, default=1)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=512)
    return parser.parse_args()


def frozen_test_rows(args, tokenizer):
    split_args = argparse.Namespace(
        data=args.data,
        train_samples=30_000,
        valid_samples=2_000,
        test_samples=2_000,
        max_scan=args.max_scan,
        seed=args.data_seed,
        source_col=args.source_col,
        target_col=args.target_col,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    rows, sampling = adaptive.load_rows(split_args, tokenizer)
    test = rows[-2_000:]
    boundaries = [(8, 12), (13, 16), (17, 24), (25, 32)]
    buckets = {f"{low}-{high}": [] for low, high in boundaries}
    leftovers = []
    for row in test:
        length = len(row[0]) - 1
        found = False
        for low, high in boundaries:
            if low <= length <= high:
                buckets[f"{low}-{high}"].append(row)
                found = True
                break
        if not found:
            leftovers.append(row)
    quota = args.samples // len(boundaries)
    chosen = []
    selected_counts = {}
    used = set()
    for key, values in buckets.items():
        take = min(quota, len(values))
        selected_counts[key] = take
        chosen.extend(values[:take])
        used.update(id(row) for row in values[:take])
    if len(chosen) < args.samples:
        pool = [row for row in test if id(row) not in used] + leftovers
        chosen.extend(pool[: args.samples - len(chosen)])
    if len(chosen) != args.samples:
        raise RuntimeError(f"selected {len(chosen)} rows; need {args.samples}")
    return chosen, {"sampling": sampling, "selected_by_length": selected_counts}


def stratified_generation_indices(rows, requested: int):
    boundaries = [(8, 12), (13, 16), (17, 24), (25, 32)]
    buckets = {f"{low}-{high}": [] for low, high in boundaries}
    for index, row in enumerate(rows):
        length = len(row[0]) - 1
        for low, high in boundaries:
            if low <= length <= high:
                buckets[f"{low}-{high}"].append(index)
                break
    quota = requested // len(boundaries)
    selected = []
    counts = {}
    for key, values in buckets.items():
        take = min(quota, len(values))
        selected.extend(values[:take])
        counts[key] = take
    if len(selected) < requested:
        used = set(selected)
        selected.extend(
            index for index in range(len(rows)) if index not in used
        )
        selected = selected[:requested]
    return set(selected), counts


def teacher_trace(decoder, levels, masks, target, bos: int, route_mode: str):
    hidden = levels[0].new_zeros((levels[0].shape[0], decoder.hidden))
    previous = torch.full(
        (levels[0].shape[0],), bos, device=levels[0].device, dtype=torch.long,
    )
    logits, routes, contexts = [], [], []
    for step in range(target.shape[1]):
        context, route = decoder.read(hidden, levels, masks, route_mode)
        hidden = decoder.cell(
            torch.cat((decoder.embedding(previous), context), dim=-1), hidden,
        )
        logits.append(decoder.output(torch.cat((hidden, context), dim=-1)))
        routes.append(route)
        contexts.append(context)
        previous = target[:, step]
    return (
        torch.stack(logits, dim=1),
        torch.stack(routes).mean(0),
        torch.stack(contexts, dim=1),
    )


def empty_metric():
    return {
        "tokens": 0,
        "nll_sum": 0.0,
        "shuffle_nll_sum": 0.0,
        "entropy_sum": 0.0,
        "top10_mass_sum": 0.0,
        "rank_sum": 0.0,
        "top1": 0,
        "top5": 0,
        "js_sum": 0.0,
        "js_tokens": 0,
        "context_l2_sum": 0.0,
        "context_cos_sum": 0.0,
        "context_tokens": 0,
        "route_sum": None,
        "route_weight": 0,
    }


def token_statistics(logits, target, pad: int):
    mask = target.ne(pad)
    count = int(mask.sum())
    log_probability = F.log_softmax(logits.float(), dim=-1)
    probability = log_probability.exp()
    safe_target = target.masked_fill(~mask, 0)
    target_log_probability = log_probability.gather(
        -1, safe_target[:, :, None],
    ).squeeze(-1)
    nll_by_token = -target_log_probability
    target_logit = logits.gather(-1, safe_target[:, :, None]).squeeze(-1)
    rank = 1 + logits.gt(target_logit[:, :, None]).sum(-1)
    top5 = logits.topk(5, dim=-1).indices
    entropy = -(probability * log_probability).sum(-1)
    top10_mass = probability.topk(10, dim=-1).values.sum(-1)
    per_example_tokens = mask.sum(1).clamp_min(1)
    return {
        "mask": mask,
        "count": count,
        "log_probability": log_probability,
        "nll_sum": float(nll_by_token.masked_select(mask).sum()),
        "entropy_sum": float(entropy.masked_select(mask).sum()),
        "top10_mass_sum": float(top10_mass.masked_select(mask).sum()),
        "rank_sum": float(rank.masked_select(mask).float().sum()),
        "top1": int(logits.argmax(-1).eq(target).logical_and(mask).sum()),
        "top5": int(top5.eq(safe_target[:, :, None]).any(-1).logical_and(mask).sum()),
        "per_example_nll": (
            (nll_by_token * mask).sum(1) / per_example_tokens
        ).detach().cpu().tolist(),
        "per_example_rank": (
            (rank.float() * mask).sum(1) / per_example_tokens
        ).detach().cpu().tolist(),
    }


def update_metric(metric, stats, shuffle_nll_sum, route, batch_size):
    metric["tokens"] += stats["count"]
    metric["nll_sum"] += stats["nll_sum"]
    metric["shuffle_nll_sum"] += shuffle_nll_sum
    metric["entropy_sum"] += stats["entropy_sum"]
    metric["top10_mass_sum"] += stats["top10_mass_sum"]
    metric["rank_sum"] += stats["rank_sum"]
    metric["top1"] += stats["top1"]
    metric["top5"] += stats["top5"]
    route_cpu = route.detach().float().cpu() * batch_size
    metric["route_sum"] = (
        route_cpu if metric["route_sum"] is None else metric["route_sum"] + route_cpu
    )
    metric["route_weight"] += batch_size


def adjacent_statistics(metric, current_stats, previous_log_probability,
                        current_context, previous_context):
    mask = current_stats["mask"]
    current_log_probability = current_stats["log_probability"]
    current_probability = current_log_probability.exp()
    previous_probability = previous_log_probability.exp()
    mixture = 0.5 * (current_probability + previous_probability)
    log_mixture = mixture.clamp_min(1e-12).log()
    js = 0.5 * (
        (current_probability * (current_log_probability - log_mixture)).sum(-1)
        + (previous_probability * (previous_log_probability - log_mixture)).sum(-1)
    )
    difference = (current_context - previous_context).square().sum(-1).sqrt()
    cosine = F.cosine_similarity(current_context, previous_context, dim=-1)
    metric["js_sum"] += float(js.masked_select(mask).sum())
    metric["js_tokens"] += int(mask.sum())
    metric["context_l2_sum"] += float(difference.masked_select(mask).sum())
    metric["context_cos_sum"] += float(cosine.masked_select(mask).sum())
    metric["context_tokens"] += int(mask.sum())


def state_accumulator(depths: int, dim: int):
    return [{
        "count": 0,
        "norm_sum": 0.0,
        "sum": torch.zeros(dim, dtype=torch.float64),
        "square_sum": torch.zeros(dim, dtype=torch.float64),
        "sibling_l2_sum": 0.0,
        "sibling_cos_sum": 0.0,
        "sibling_count": 0,
        "parent_child_l2_sum": 0.0,
        "parent_child_cos_sum": 0.0,
        "parent_child_count": 0,
    } for _ in range(depths)]


def update_state_metrics(accumulators, levels, masks):
    for depth, (nodes, valid) in enumerate(zip(levels, masks)):
        selected = nodes[valid]
        accumulator = accumulators[depth]
        accumulator["count"] += selected.shape[0]
        accumulator["norm_sum"] += float(selected.float().norm(dim=-1).sum())
        accumulator["sum"] += selected.double().sum(0).cpu()
        accumulator["square_sum"] += selected.double().square().sum(0).cpu()
        if depth == 0:
            continue
        pair_valid = valid[:, 0::2] & valid[:, 1::2]
        left, right = nodes[:, 0::2], nodes[:, 1::2]
        if bool(pair_valid.any()):
            l, r = left[pair_valid], right[pair_valid]
            accumulator["sibling_l2_sum"] += float((l - r).float().norm(dim=-1).sum())
            accumulator["sibling_cos_sum"] += float(F.cosine_similarity(l.float(), r.float(), dim=-1).sum())
            accumulator["sibling_count"] += l.shape[0]
        parent = levels[depth - 1]
        parent_repeated = parent[:, :, None, :].expand(-1, -1, 2, -1).reshape_as(nodes)
        child_selected = nodes[valid]
        parent_selected = parent_repeated[valid]
        accumulator["parent_child_l2_sum"] += float(
            (child_selected - parent_selected).float().norm(dim=-1).sum()
        )
        accumulator["parent_child_cos_sum"] += float(
            F.cosine_similarity(child_selected.float(), parent_selected.float(), dim=-1).sum()
        )
        accumulator["parent_child_count"] += child_selected.shape[0]


def finalize_state_metrics(accumulators):
    rows = []
    for depth, accumulator in enumerate(accumulators):
        count = max(1, accumulator["count"])
        mean = accumulator["sum"] / count
        variance = (accumulator["square_sum"] / count - mean.square()).clamp_min(0)
        sibling_count = max(1, accumulator["sibling_count"])
        parent_count = max(1, accumulator["parent_child_count"])
        rows.append({
            "depth": depth,
            "valid_nodes": accumulator["count"],
            "mean_state_norm": accumulator["norm_sum"] / count,
            "mean_dimension_variance": float(variance.mean()),
            "sibling_l2": accumulator["sibling_l2_sum"] / sibling_count,
            "sibling_cosine": accumulator["sibling_cos_sum"] / sibling_count,
            "parent_child_l2": accumulator["parent_child_l2_sum"] / parent_count,
            "parent_child_cosine": accumulator["parent_child_cos_sum"] / parent_count,
        })
    return rows


def finalize_metric(metric):
    tokens = max(1, metric["tokens"])
    route = metric["route_sum"] / max(1, metric["route_weight"])
    nll = metric["nll_sum"] / tokens
    shuffle_nll = metric["shuffle_nll_sum"] / tokens
    entropy = metric["entropy_sum"] / tokens
    return {
        "nll": nll,
        "ppl": math.exp(min(20.0, nll)),
        "source_shuffle_nll": shuffle_nll,
        "source_shuffle_damage_nll": shuffle_nll - nll,
        "target_mean_rank": metric["rank_sum"] / tokens,
        "target_top1": metric["top1"] / tokens,
        "target_top5": metric["top5"] / tokens,
        "entropy_nats": entropy,
        "effective_candidates_exp_entropy": math.exp(min(20.0, entropy)),
        "top10_mass": metric["top10_mass_sum"] / tokens,
        "adjacent_js": metric["js_sum"] / max(1, metric["js_tokens"]),
        "adjacent_context_l2": metric["context_l2_sum"] / max(1, metric["context_tokens"]),
        "adjacent_context_cosine": metric["context_cos_sum"] / max(1, metric["context_tokens"]),
        "route_mass": route.tolist(),
        "tokens": metric["tokens"],
    }


def load_decoders(args, vocab: int, depths: int):
    decoders = {}
    metadata = {}
    for arm, route_mode in ARMS.items():
        path = Path(args.decoder_dir) / f"decoder_{arm}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        decoder = c06.FloorPressureDecoder(
            vocab, args.dim, args.hidden, depths, c06.DEPTH_FLOOR,
        )
        decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
        decoder.to(args.device).eval()
        decoders[arm] = (decoder, route_mode)
        metadata[arm] = {
            "path": str(path),
            "declared_arm": payload.get("arm"),
            "declared_route_mode": payload.get("route_mode"),
            "source_checkpoint": payload.get("source_checkpoint"),
        }
    return decoders, metadata


@torch.inference_mode()
def run(args):
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces = tokenizer.get_piece_size()
    pad, vocab = pieces, pieces + 1
    rows, sampling = frozen_test_rows(args, tokenizer)
    generation_indices, generation_counts = stratified_generation_indices(
        rows, min(args.generation_samples, len(rows)),
    )
    loader = DataLoader(
        base.ParallelDataset(rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=base.collate(pad),
        pin_memory=args.device.startswith("cuda"),
    )
    model_args = argparse.Namespace(
        c04_checkpoint=args.c04_checkpoint,
        dim=args.dim,
        hidden=args.hidden,
        heap_width=args.heap_width,
        leaf_cut=args.leaf_cut,
    )
    shared_model = c06.load_model(model_args, vocab, pad).to(args.device).eval()
    decoders, decoder_metadata = load_decoders(
        args, vocab, shared_model.encoder.depths,
    )
    encoder_digest = c05.tensor_digest(shared_model.encoder)
    visible_depths = shared_model.encoder.depths + 1 - args.leaf_cut
    metrics = {
        arm: [empty_metric() for _ in range(visible_depths)]
        for arm in ARMS
    }
    state_metrics = state_accumulator(visible_depths, args.dim)
    per_example_path = output / "per_example.jsonl"
    if per_example_path.exists():
        per_example_path.unlink()
    example_index = 0

    for batch_index, (source, length, target, _) in enumerate(loader):
        source = source.to(args.device, non_blocking=True)
        length = length.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        state = shared_model.states(source, length)
        all_levels, all_masks = shared_model.visible(state[3], state[4])
        update_state_metrics(state_metrics, all_levels, all_masks)
        local_rows = []
        for row_index in range(source.shape[0]):
            source_tokens = base.clean(source[row_index].cpu().tolist(), tokenizer.eos_id(), pad)
            target_tokens = base.clean(target[row_index].cpu().tolist(), tokenizer.eos_id(), pad)
            local_rows.append({
                "index": example_index + row_index,
                "source_length": int(length[row_index]),
                "source": tokenizer.decode(source_tokens),
                "reference": tokenizer.decode(target_tokens),
                "observations": {},
            })

        for arm, (decoder, route_mode) in decoders.items():
            previous_log_probability = None
            previous_context = None
            for visible_count in range(1, visible_depths + 1):
                depth = visible_count - 1
                levels = all_levels[:visible_count]
                masks = all_masks[:visible_count]
                logits, route, contexts = teacher_trace(
                    decoder, levels, masks, target, tokenizer.bos_id(), route_mode,
                )
                stats = token_statistics(logits, target, pad)
                shuffled_levels = [level.roll(1, dims=0) for level in levels]
                shuffled_masks = [mask.roll(1, dims=0) for mask in masks]
                shuffled_logits, _, _ = teacher_trace(
                    decoder, shuffled_levels, shuffled_masks, target,
                    tokenizer.bos_id(), route_mode,
                )
                shuffle_nll_sum = float(F.cross_entropy(
                    shuffled_logits.reshape(-1, shuffled_logits.shape[-1]),
                    target.reshape(-1), ignore_index=pad, reduction="sum",
                ))
                metric = metrics[arm][depth]
                update_metric(metric, stats, shuffle_nll_sum, route, source.shape[0])
                if previous_log_probability is not None:
                    adjacent_statistics(
                        metric, stats, previous_log_probability,
                        contexts, previous_context,
                    )
                previous_log_probability = stats["log_probability"]
                previous_context = contexts

                generated_by_local_index = {}
                selected_local = [
                    row_index for row_index in range(source.shape[0])
                    if example_index + row_index in generation_indices
                ]
                if selected_local:
                    selected_tensor = torch.tensor(
                        selected_local, dtype=torch.long, device=args.device,
                    )
                    predicted, _ = decoder.greedy(
                        [level.index_select(0, selected_tensor) for level in levels],
                        [mask.index_select(0, selected_tensor) for mask in masks],
                        tokenizer.bos_id(), tokenizer.eos_id(),
                        args.generation_length, route_mode,
                    )
                    generated_cpu = predicted.cpu()
                    generated_by_local_index = {
                        local_index: generated_cpu[position]
                        for position, local_index in enumerate(selected_local)
                    }

                for row_index, row in enumerate(local_rows):
                    key = f"{arm}:D{depth}"
                    observation = {
                        "teacher_nll": stats["per_example_nll"][row_index],
                        "target_mean_rank": stats["per_example_rank"][row_index],
                    }
                    if row_index in generated_by_local_index:
                        tokens = base.clean(
                            generated_by_local_index[row_index].tolist(),
                            tokenizer.eos_id(), pad,
                        )
                        observation["generation"] = tokenizer.decode(tokens)
                        observation["severe_repetition"] = c01.severe_repetition(tokens)
                    row["observations"][key] = observation

        with per_example_path.open("a", encoding="utf-8") as handle:
            for row in local_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        example_index += source.shape[0]
        print(json.dumps({
            "event": "batch_complete",
            "batch": batch_index + 1,
            "examples": example_index,
            "elapsed_sec": time.time() - started,
        }), flush=True)

    result = {
        "kind": "matched_decoder_resolution_observation",
        "training": False,
        "claim_registered": False,
        "device": args.device,
        "samples": len(rows),
        "generation_samples": len(generation_indices),
        "generation_selected_by_length": generation_counts,
        "sampling": sampling,
        "encoder": {
            "path": args.c04_checkpoint,
            "sha256_state_dict": encoder_digest,
            "heap_width": args.heap_width,
            "visible_depths": visible_depths,
        },
        "decoders": decoder_metadata,
        "state_by_depth": finalize_state_metrics(state_metrics),
        "decoder_by_depth": {
            arm: [finalize_metric(metric) for metric in rows]
            for arm, rows in metrics.items()
        },
        "per_example": str(per_example_path),
        "elapsed_sec": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "event": "complete",
        "summary": str(output / "summary.json"),
        "elapsed_sec": result["elapsed_sec"],
    }), flush=True)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
