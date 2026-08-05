#!/usr/bin/env python3
"""C07 matched product-recipe selector for the bilingual Butterfly TreeHeap."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3_stone1_private_protocol as stone1  # noqa: E402
import s3_treeheap_butterfly_bilingual_full as base  # noqa: E402
import s3_treeheap_canonical_view_dose as c06  # noqa: E402
import s3_treeheap_canonical_view_ratio as c05  # noqa: E402
import s3_wmt_treeheap_seq2seq as wmtseq  # noqa: E402


CLAIM = "S3-TREEHEAP-BUTTERFLY-PRODUCT-C07"
ARMS = ("N_native", "BB_replay", "BI_replay")
ARM_KERNEL = {
    "N_native": "A_base",
    "BB_replay": "BB_add_butterfly",
    "BI_replay": "BI_add_identity",
}
SEEDS = (9201, 9202, 9203)
EXPECTED = {
    "checkpoint": "821ce8123d78817b37ff8f0a68458fd59427a7af555f93c7c87c297f28861c1d",
    "data": "3f4a5189a6b2f06a8a928165a69e119d6e0afe71ffece2bbe7c049ecef7a44df",
    "spm_model": "9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731",
    "dreams_core": "049d81bdb6400793b489e3c2ee89d77870933465ff9a8bd7addd4f1957cf8349",
}


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def preflight(args) -> dict:
    paths = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "spm_model": args.spm_model,
        "dreams_core": args.dreams_core,
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED[name], "actual": value}
        for name, value in hashes.items() if value != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(f"C07 preflight hash mismatch: {mismatches}")
    return {
        "claim": CLAIM,
        "hashes": hashes,
        "sizes": {name: Path(path).stat().st_size for name, path in paths.items()},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": args.device,
        "time": time.time(),
    }


def parse_core_dreams(path: Path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_no}: expected 6 TSV fields")
        probe_id, direction, category, source, reference, facts = parts
        if direction not in base.DIRECTIONS:
            raise ValueError(f"{path}:{line_no}: invalid direction {direction}")
        rows.append({
            "id": probe_id,
            "direction": direction,
            "category": category,
            "source": source,
            "reference": reference,
            "required_facts": facts,
        })
    if len(rows) != 32 or len({row["id"] for row in rows}) != 32:
        raise ValueError("C07 core Dreams require 32 unique probes")
    return rows


def make_context(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    pieces, eos, bos = sp.get_piece_size(), sp.eos_id(), sp.bos_id()
    pad = pieces
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    vocab = pieces + 3
    eval_rows = base.read_eval_rows(args, sp, direction_ids, eos)
    old_dreams = base.parse_dreams(Path(args.dreams))
    grammar_rows = c05.reference_dream_rows(
        old_dreams, sp, direction_ids, eos, args,
    )
    core_dreams = parse_core_dreams(Path(args.dreams_core))
    return checkpoint, sp, pieces, eos, bos, pad, direction_ids, vocab, eval_rows, grammar_rows, old_dreams, core_dreams


def load_arm_model(path: Path, checkpoint, args, vocab: int, pad: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = c05.make_model(checkpoint, args, vocab, pad)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def clean_output(tokens, eos: int, pieces: int):
    return base.clean(tokens, eos, pieces)


def length_bucket(length: int) -> str:
    if length <= 32:
        return "le32"
    if length <= 64:
        return "33_64"
    if length <= 128:
        return "65_128"
    return "129_256"


@torch.no_grad()
def generation_metrics(model, rows, args, sp, pad, bos, eos, pieces):
    chosen = rows[: args.generation_pairs * 2]
    hypotheses = defaultdict(list)
    references = defaultdict(list)
    hyp_tokens = defaultdict(list)
    ref_tokens = defaultdict(list)
    repetitions = defaultdict(int)
    nonempty = defaultdict(int)
    buckets = defaultdict(lambda: {"rows": 0, "nonempty": 0, "repeated": 0})
    examples = []
    rng = random.Random(1701)
    for batch in base.rows_to_batches(chosen[:], args, rng):
        source, length, target = base.collate(batch, pad, args.device)
        maximum = min(args.generation_max_output, max(8, int(target.shape[1]) + 16))
        generated, _route = model.greedy(source, length, bos, eos, maximum)
        generated = generated.cpu()
        target = target.cpu()
        length_cpu = length.cpu()
        for index, row in enumerate(batch):
            direction = row[2]
            hyp = clean_output(generated[index].tolist(), eos, pieces)
            ref = clean_output(target[index].tolist(), eos, pieces)
            hyp_text, ref_text = sp.decode(hyp), sp.decode(ref)
            repeated = stone1.severe_repetition(hyp)
            hypotheses[direction].append(hyp_text)
            references[direction].append(ref_text)
            hyp_tokens[direction].append(hyp)
            ref_tokens[direction].append(ref)
            repetitions[direction] += int(repeated)
            nonempty[direction] += int(bool(hyp))
            bucket = length_bucket(int(length_cpu[index]))
            buckets[bucket]["rows"] += 1
            buckets[bucket]["nonempty"] += int(bool(hyp))
            buckets[bucket]["repeated"] += int(repeated)
            if len(examples) < 24:
                examples.append({
                    "direction": direction,
                    "source": row[3][1] if direction == "en2zh" else row[3][0],
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "source_pieces": int(length_cpu[index]),
                    "severe_repetition": repeated,
                })
    per_direction = {}
    for direction in base.DIRECTIONS:
        count = len(hypotheses[direction])
        tokenizer = "zh" if direction == "en2zh" else "13a"
        per_direction[direction] = {
            "rows": count,
            "chrf2": sacrebleu.corpus_chrf(
                hypotheses[direction], [references[direction]], word_order=2,
            ).score,
            "sacrebleu": sacrebleu.corpus_bleu(
                hypotheses[direction], [references[direction]], tokenize=tokenizer,
            ).score,
            "token_bleu4": wmtseq.bleu4(
                hyp_tokens[direction], ref_tokens[direction],
            ),
            "nonempty_rate": nonempty[direction] / max(1, count),
            "severe_repetition_rate": repetitions[direction] / max(1, count),
            "distinct_output_rate": len(set(hypotheses[direction])) / max(1, count),
        }
    bucket_result = {}
    for name, values in buckets.items():
        bucket_result[name] = {
            **values,
            "nonempty_rate": values["nonempty"] / max(1, values["rows"]),
            "severe_repetition_rate": values["repeated"] / max(1, values["rows"]),
        }
    return {
        "per_direction": per_direction,
        "chrf2_mean": statistics.fmean(row["chrf2"] for row in per_direction.values()),
        "sacrebleu_mean": statistics.fmean(row["sacrebleu"] for row in per_direction.values()),
        "token_bleu4_mean": statistics.fmean(row["token_bleu4"] for row in per_direction.values()),
        "severe_repetition_mean": statistics.fmean(row["severe_repetition_rate"] for row in per_direction.values()),
        "nonempty_min": min(row["nonempty_rate"] for row in per_direction.values()),
        "buckets": bucket_result,
        "examples": examples,
    }


@torch.no_grad()
def render_core_dreams(model, dreams, output: Path, args, sp, direction_ids, pad, bos, eos, pieces):
    records = []
    for row in dreams:
        source_ids = [
            direction_ids[row["direction"]],
            *sp.encode(row["source"], out_type=int),
            eos,
        ]
        source = torch.tensor([source_ids], device=args.device)
        length = torch.tensor([len(source_ids)], device=args.device)
        generated, route = model.greedy(
            source, length, bos, eos, args.generation_max_output,
        )
        hypothesis_ids = clean_output(generated[0].cpu().tolist(), eos, pieces)
        records.append({
            **row,
            "source_raw_pieces": len(source_ids) - 2,
            "hypothesis": sp.decode(hypothesis_ids),
            "hypothesis_ids": hypothesis_ids,
            "severe_repetition": stone1.severe_repetition(hypothesis_ids),
            "route": [float(value) for value in route.cpu()],
        })
    path = output / "core_dreams.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return {"path": str(path), "rows": len(records), "sha256": digest(path)}


def run_arm(args):
    if args.arm not in ARMS:
        raise ValueError(f"arm required for action=arm: {args.arm}")
    if args.require_smoke and not Path(args.smoke_marker).is_file():
        raise RuntimeError(f"C07 smoke marker missing: {args.smoke_marker}")
    environment = preflight(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    context = make_context(args)
    checkpoint, sp, pieces, eos, bos, pad, direction_ids, vocab, eval_rows, grammar_rows, old_dreams, core_dreams = context
    arm_root = Path(args.evidence_dir) / f"selector_seed{args.seed}"
    local_args = argparse.Namespace(**vars(args))
    local_args.evidence_dir = str(arm_root)
    local_args.batch_seed = args.seed
    local_args.save_arm_checkpoints = True
    local_args.resume_completed = False
    kernel = ARM_KERNEL[args.arm]
    summary = c06.train_arm(
        checkpoint, kernel, local_args, sp, direction_ids, pad, bos, eos, vocab,
        eval_rows["valid"], grammar_rows, old_dreams,
    )
    output = arm_root / f"{kernel}_seed{args.seed}"
    checkpoint_path = output / "checkpoint_final.pt"
    model = load_arm_model(checkpoint_path, checkpoint, args, vocab, pad)
    product = generation_metrics(
        model, eval_rows["valid"], args, sp, pad, bos, eos, pieces,
    )
    dream_result = render_core_dreams(
        model, core_dreams, output, args, sp, direction_ids,
        pad, bos, eos, pieces,
    )
    summary["parent_claim"] = CLAIM
    summary["product_arm"] = args.arm
    summary["environment"] = environment
    summary["product_generation"] = product
    summary["core_dreams"] = dream_result
    summary["checkpoint_final"] = {
        "path": str(checkpoint_path),
        "sha256": digest(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "event": "c07_arm_finished", "arm": args.arm, "seed": args.seed,
        "native_nll": summary["metrics"]["native"]["mean"],
        "cross_view_js": summary["metrics"]["cross_view_js"],
        "chrf2": product["chrf2_mean"],
        "bleu": product["sacrebleu_mean"],
        "repetition": product["severe_repetition_mean"],
    }, ensure_ascii=False), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_smoke(args):
    root = Path(args.evidence_dir)
    smoke_root = root / "smoke"
    if smoke_root.exists() and any(smoke_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite smoke evidence: {smoke_root}")
    smoke_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        local = argparse.Namespace(**vars(args))
        local.action = "arm"
        local.arm = arm
        local.seed = 9199
        local.batch_seed = 9199
        local.start_line = 6_700_000
        local.train_lines = args.smoke_lines
        local.block_lines = args.smoke_lines
        local.eval_pairs = args.smoke_eval_pairs
        local.generation_pairs = args.smoke_generation_pairs
        local.generation_max_output = min(64, args.generation_max_output)
        local.evidence_dir = str(smoke_root)
        local.require_smoke = False
        results[arm] = run_arm(local)
    updates = {arm: row["updates"] for arm, row in results.items()}
    lrs = {arm: row["effective_lr"] for arm, row in results.items()}
    extra_match = (
        results["BB_replay"]["counts"]["extra_tokens"]
        == results["BI_replay"]["counts"]["extra_tokens"]
        and results["BB_replay"]["counts"]["extra_examples"]
        == results["BI_replay"]["counts"]["extra_examples"]
    )
    gates = {
        "one_step_updates_match": len(set(updates.values())) == 1,
        "effective_lr_exact": all(
            all(abs(value - args.lr) < 1e-12 for value in values)
            for values in lrs.values()
        ),
        "extra_dose_match": extra_match,
        "all_metrics_finite": all(
            math.isfinite(row["metrics"]["native"]["mean"])
            and math.isfinite(row["product_generation"]["chrf2_mean"])
            for row in results.values()
        ),
        "all_checkpoints_saved": all(
            Path(row["checkpoint_final"]["path"]).is_file()
            for row in results.values()
        ),
    }
    report = {
        "claim": CLAIM, "status": "pass" if all(gates.values()) else "fail",
        "updates": updates, "effective_lr": lrs, "gates": gates,
        "results": {
            arm: {
                "native_nll": row["metrics"]["native"]["mean"],
                "cross_view_js": row["metrics"]["cross_view_js"],
                "chrf2": row["product_generation"]["chrf2_mean"],
                "repetition": row["product_generation"]["severe_repetition_mean"],
            } for arm, row in results.items()
        },
    }
    report_path = root / "smoke_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(gates.values()):
        raise RuntimeError(f"C07 smoke failed: {gates}")
    Path(args.smoke_marker).write_text(
        json.dumps({**report, "summary": str(report_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def arm_summary_path(root: Path, seed: int, arm: str) -> Path:
    kernel = ARM_KERNEL[arm]
    return root / f"selector_seed{seed}" / f"{kernel}_seed{seed}" / "summary.json"


def summarize(args):
    root = Path(args.evidence_dir)
    runs = {}
    for seed in SEEDS:
        runs[str(seed)] = {}
        for arm in ARMS:
            path = arm_summary_path(root, seed, arm)
            if not path.is_file():
                raise FileNotFoundError(path)
            runs[str(seed)][arm] = json.loads(path.read_text(encoding="utf-8"))
    per_seed = {}
    for seed in SEEDS:
        rows = runs[str(seed)]
        bb, bi = rows["BB_replay"], rows["BI_replay"]
        per_seed[str(seed)] = {
            "js_specificity": (
                bb["metrics"]["cross_view_js"] - bi["metrics"]["cross_view_js"]
            ),
            "native_cost": (
                bi["metrics"]["native"]["mean"] - bb["metrics"]["native"]["mean"]
            ),
            "chrf_difference": (
                bi["product_generation"]["chrf2_mean"]
                - bb["product_generation"]["chrf2_mean"]
            ),
            "bleu_difference": (
                bi["product_generation"]["sacrebleu_mean"]
                - bb["product_generation"]["sacrebleu_mean"]
            ),
            "repetition_increase": (
                bi["product_generation"]["severe_repetition_mean"]
                - bb["product_generation"]["severe_repetition_mean"]
            ),
            "source_shuffle_damage": bi["metrics"]["source_shuffle_damage"],
            "adjacent_damage": bi["metrics"]["adjacent_damage"],
            "communication_delta_rms": bi["metrics"]["communication_delta_rms"],
            "communication_grad_norm": bi["communication_grad_norm_mean"],
        }
    values = list(per_seed.values())
    aggregate = {
        key: statistics.fmean(row[key] for row in values)
        for key in (
            "js_specificity", "native_cost", "chrf_difference",
            "bleu_difference", "repetition_increase",
        )
    }
    gates = {
        "mean_js_specificity_ge_0p10": aggregate["js_specificity"] >= 0.10,
        "each_js_specificity_ge_0p05": all(row["js_specificity"] >= 0.05 for row in values),
        "mean_native_cost_le_0p01": aggregate["native_cost"] <= 0.01,
        "each_native_cost_le_0p02": all(row["native_cost"] <= 0.02 for row in values),
        "mean_chrf_difference_ge_neg0p30": aggregate["chrf_difference"] >= -0.30,
        "mean_bleu_difference_ge_neg0p20": aggregate["bleu_difference"] >= -0.20,
        "max_repetition_increase_le_0p02": max(row["repetition_increase"] for row in values) <= 0.02,
        "source_shuffle_damage_ge_1p50": all(row["source_shuffle_damage"] >= 1.50 for row in values),
        "adjacent_damage_positive": all(row["adjacent_damage"] > 0 for row in values),
        "communication_causal": all(
            row["communication_delta_rms"] > 0 and row["communication_grad_norm"] > 0
            for row in values
        ),
    }
    selected = "BI_replay" if all(gates.values()) else "N_native"
    summary = {
        "claim": CLAIM,
        "status": "selector_complete",
        "selected_product_recipe": selected,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "boundary": "Selector evidence only; full-corpus product continuation is not yet signed.",
    }
    path = root / "selector_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    base.notify(f"TreeHeap C07 selector finished: {selected}", path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=("smoke", "arm", "summarize"), required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--checkpoint", default="ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt")
    p.add_argument("--data", default="/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv")
    p.add_argument("--spm-model", default="/home/nio/datasets/wmt_massive/sp_bpe_massive.model")
    p.add_argument("--dreams", default="ara/s3-generation/dreams.txt")
    p.add_argument("--dreams-core", default="ara/s3-generation/dreams_product_v1_core.tsv")
    p.add_argument("--evidence-dir", default="ara/s3-generation/evidence/s3_stone1_butterfly_product_v1")
    p.add_argument("--smoke-marker", default="ara/s3-generation/evidence/s3_stone1_butterfly_product_v1/SMOKE_PASS")
    p.add_argument("--require-smoke", action="store_true")
    p.add_argument("--seed", type=int, default=9201)
    p.add_argument("--start-line", type=int, default=6_700_000)
    p.add_argument("--train-lines", type=int, default=1_000_000)
    p.add_argument("--block-lines", type=int, default=100_000)
    p.add_argument("--smoke-lines", type=int, default=30_000)
    p.add_argument("--smoke-eval-pairs", type=int, default=64)
    p.add_argument("--smoke-generation-pairs", type=int, default=16)
    p.add_argument("--eval-pairs", type=int, default=1_000)
    p.add_argument("--eval-scan", type=int, default=3_000_000)
    p.add_argument("--generation-pairs", type=int, default=128)
    p.add_argument("--generation-max-output", type=int, default=128)
    p.add_argument("--diagnostic-rows", type=int, default=128)
    p.add_argument("--diagnostic-batch", type=int, default=4)
    p.add_argument("--dream-limit", type=int, default=0)
    p.add_argument("--dream-max-output", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-seed", type=int, default=9201)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reuse-optimizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-completed", action="store_true")
    p.add_argument("--save-arm-checkpoints", action="store_true", default=True)
    p.add_argument("--heap-width", type=int, default=256)
    p.add_argument("--max-content", type=int, default=253)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--coupling-scale", type=float, default=0.25)
    p.add_argument("--batch-32", type=int, default=64)
    p.add_argument("--batch-64", type=int, default=32)
    p.add_argument("--batch-128", type=int, default=16)
    p.add_argument("--batch-256", type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.action == "smoke":
        run_smoke(args)
    elif args.action == "arm":
        run_arm(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
