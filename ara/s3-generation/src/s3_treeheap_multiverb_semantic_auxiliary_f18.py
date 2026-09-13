#!/usr/bin/env python3
"""Train a shared leaf/root target-presence head while preserving translation loss."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import s3_pretrain_task_posterior_pipeline as c10
import s3_recursive_depth_pressure_protocol_training as d07
import s3_structural_protocol_capacity_ladder_d11 as d11
import s3_structural_protocol_full_pipeline_d10 as d10
import s3_treeheap_cross_language_semantic_recoverability_f16 as f16
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-MULTIVERB-SEMANTIC-AUXILIARY-F18"
CJK = re.compile(r"[\u4e00-\u9fff]")
CONCEPTS = {
    "push": {
        "target": re.compile(r"(?i)\bpush(?:es|ed|ing)?\b"),
        "families": (
            ("physical", r"推着|推开|推入|推倒|推向|推过|推车|推石|推门|推球|推人"),
            ("technical", r"推送|推流|版本库"),
            ("abstract", r"推动|推进|推广|推行|推出|推迟"),
            ("press", r"按|压"),
            ("force", r"迫使|逼|驱使|强迫"),
            ("other_tui", r"推"),
        ),
    },
    "like": {
        "target": re.compile(r"(?i)\blike(?:s|d|ing)?\b"),
        "families": (
            ("preference", r"喜欢"), ("fondness", r"喜爱|爱好"),
            ("love", r"爱"), ("similarity", r"像|如同|类似"),
        ),
    },
    "want": {
        "target": re.compile(r"(?i)\bwant(?:s|ed|ing)?\b"),
        "families": (
            ("direct", r"想要"), ("hope", r"希望"), ("desire", r"渴望|愿望"),
            ("intend", r"打算|想"), ("need", r"要"),
        ),
    },
    "eat": {
        "target": re.compile(r"(?i)\b(?:eat|eats|ate|eaten|eating)\b"),
        "families": (
            ("direct", r"吃"), ("consume", r"食用|进食"),
            ("swallow", r"吞"), ("meal", r"用餐|就餐"),
        ),
    },
    "carry": {
        "target": re.compile(r"(?i)\b(?:carry|carries|carried|carrying)\b"),
        "families": (
            ("direct", r"携带"), ("move", r"搬|运输"),
            ("shoulder", r"扛|背着"), ("bring", r"带着|带上"),
        ),
    },
}
CONCEPT_NAMES = tuple(CONCEPTS)
for spec in CONCEPTS.values():
    spec["families"] = tuple((name, re.compile(pattern)) for name, pattern in spec["families"])


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stable_split(line_number: int) -> str:
    digest = hashlib.sha256(f"f18-split:{line_number}".encode("ascii")).digest()
    return "eval" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def source_family(concept: str, text: str) -> str | None:
    for family, pattern in CONCEPTS[concept]["families"]:
        if pattern.search(text):
            return family
    return None


def target_labels(text: str) -> list[float]:
    return [float(bool(CONCEPTS[name]["target"].search(text))) for name in CONCEPT_NAMES]


def collect_candidates(path: Path, excluded: set[int], cap: int, seed: int):
    keys = [
        (concept, family, label, split)
        for concept, spec in CONCEPTS.items()
        for family, _ in spec["families"]
        for label in (0, 1) for split in ("train", "eval")
    ]
    buckets = {key: [] for key in keys}
    seen = collections.Counter()
    rng = random.Random(seed)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in excluded:
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 2:
                continue
            zh, en = columns
            if len(CJK.findall(zh)) < 4 or "�" in zh or "�" in en:
                continue
            split = stable_split(line_number)
            for concept in CONCEPT_NAMES:
                family = source_family(concept, zh)
                if family is None:
                    continue
                label = int(bool(CONCEPTS[concept]["target"].search(en)))
                key = (concept, family, label, split)
                seen[key] += 1
                row = {
                    "line": line_number, "zh": zh, "en": en, "concept": concept,
                    "family": family, "focal_label": label, "split": split,
                }
                bucket = buckets[key]
                if len(bucket) < cap:
                    bucket.append(row)
                else:
                    replacement = rng.randrange(seen[key])
                    if replacement < cap:
                        bucket[replacement] = row
    return buckets, dict(seen)


def pair_by_length(positives, negatives, wanted: int, identity: str):
    positives = sorted(positives, key=lambda row: (row["source_pieces"], row["line"]))
    negatives = list(negatives)
    pairs, used = [], set()
    for positive in positives:
        choices = [
            (abs(negative["source_pieces"] - positive["source_pieces"]), negative["line"], index)
            for index, negative in enumerate(negatives) if index not in used
        ]
        if not choices:
            break
        _, _, index = min(choices)
        used.add(index)
        pairs.append((positive, negatives[index]))
    pairs.sort(key=lambda pair: hashlib.sha256(
        f"{identity}:{pair[0]['line']}:{pair[1]['line']}".encode("utf-8")
    ).digest())
    if len(pairs) < wanted:
        raise ValueError(f"insufficient length-matched pairs for {identity}: {len(pairs)}/{wanted}")
    return pairs[:wanted]


def build_manifest(buckets, sp, train_per_class: int, eval_per_class: int):
    selected = []
    for concept, spec in CONCEPTS.items():
        for family, _ in spec["families"]:
            for split, wanted in (("train", train_per_class), ("eval", eval_per_class)):
                by_label = {}
                for label in (0, 1):
                    valid = []
                    for row in buckets[(concept, family, label, split)]:
                        source_ids = sp.encode(row["zh"], out_type=int)
                        target_ids = sp.encode(row["en"], out_type=int)
                        if 2 <= len(source_ids) <= 30 and target_ids:
                            valid.append({
                                **row, "source_ids": source_ids, "target_ids": target_ids[:31],
                                "source_pieces": len(source_ids),
                                "target_pieces": min(len(target_ids), 31),
                                "labels": target_labels(row["en"]),
                            })
                    by_label[label] = valid
                pairs = pair_by_length(
                    by_label[1], by_label[0], wanted, f"{concept}:{family}:{split}",
                )
                for pair_index, (positive, negative) in enumerate(pairs):
                    for row in (positive, negative):
                        selected.append({
                            **row,
                            "id": f"{concept}:{family}:{split}:{pair_index:03d}:{row['focal_label']}",
                        })
    selected.sort(key=lambda row: row["id"])
    return selected


def collate(rows, pieces: int, eos: int, device: str):
    source = torch.full((len(rows), 32), pieces, dtype=torch.long, device=device)
    source_lengths = torch.empty(len(rows), dtype=torch.long, device=device)
    target_width = max(len(row["target_ids"]) + 1 for row in rows)
    target = torch.full((len(rows), target_width), pieces, dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"] for row in rows], dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        source_ids = [pieces + 2, *row["source_ids"], eos]
        target_ids = [*row["target_ids"], eos]
        source[index, :len(source_ids)] = torch.tensor(source_ids, dtype=torch.long, device=device)
        source_lengths[index] = len(source_ids)
        target[index, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long, device=device)
    return source, source_lengths, target, labels


def masked_mean(nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(nodes.dtype)
    return (nodes * weight[:, :, None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1.0)


def forward_with_features(model, source, lengths, target, bos: int, depth: int):
    base_tree, base_masks, budgets, slots, entropy, extra = model.protocol(source, lengths, depth)
    base_logits, route = model.reconstructor.teacher(base_tree, base_masks, target, bos)
    root = masked_mean(base_tree[0], base_masks[0])
    leaf = masked_mean(base_tree[-1], base_masks[-1])
    if extra is not None:
        extra_tree, extra_masks, _, _ = extra
        extra_logits, extra_route = model.extra.reconstructor.teacher(extra_tree, extra_masks, target, bos)
        base_logits = base_logits + torch.tanh(model.extra_logit_gain) * extra_logits
        route = (route + extra_route) * 0.5
        root = torch.cat((root, masked_mean(extra_tree[0], extra_masks[0])), dim=-1)
        leaf = torch.cat((leaf, masked_mean(extra_tree[-1], extra_masks[-1])), dim=-1)
    return base_logits, root, leaf, route, budgets, slots, entropy


def batches(rows, batch_size: int, steps: int, seed: int):
    rng = random.Random(seed)
    order = list(range(len(rows)))
    cursor = len(order)
    for _ in range(steps):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor:cursor + batch_size]
        cursor += batch_size
        yield [rows[index] for index in indices]


@torch.no_grad()
def head_evaluation(model, head, rows, pieces, eos, bos, batch_size, device):
    model.eval()
    head.eval()
    result = []
    for depth in DEPTHS:
        collected = {"root": [], "leaf": []}
        labels = []
        focal = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            source, lengths, target, batch_labels = collate(batch, pieces, eos, device)
            _, root, leaf, _, _, _, _ = forward_with_features(
                model, source, lengths, target, bos, depth,
            )
            collected["root"].append(head(root).float().cpu())
            collected["leaf"].append(head(leaf).float().cpu())
            labels.append(batch_labels.cpu())
            focal.extend(row["concept"] for row in batch)
        labels = torch.cat(labels)
        for level, parts in collected.items():
            logits = torch.cat(parts)
            for concept_index, concept in enumerate(CONCEPT_NAMES):
                mask = torch.tensor([value == concept for value in focal])
                target = labels[mask, concept_index].long()
                scores = logits[mask, concept_index]
                result.append({
                    "depth": depth, "level": level, "concept": concept,
                    "examples": int(mask.sum()), "auroc": f16.auc(scores, target),
                    "average_precision": f16.average_precision(scores, target),
                    "balanced_accuracy": float(0.5 * (
                        (scores[target == 1] >= 0).double().mean()
                        + (scores[target == 0] < 0).double().mean()
                    )),
                })
    return result


@torch.no_grad()
def f16_probe(model, rows, pieces, eos, batch_size, device):
    model.eval()
    features = f16.extract_features(model, rows, pieces, eos, batch_size, device)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    output = {}
    for depth in DEPTHS:
        for stage in ("fold", "read_facing"):
            for level in ("root", "leaf"):
                values = features[depth][stage][level]
                scores = []
                for family, _ in f16.FAMILIES:
                    train_mask = torch.tensor([row["family"] != family for row in rows])
                    test_mask = ~train_mask
                    metrics = f16.fit_direction(values, labels, train_mask, test_mask)
                    scores.append(metrics["auroc"])
                output[f"d{depth}:{stage}:{level}"] = sum(scores) / len(scores)
    for stage in ("fold", "read_facing"):
        for level in ("root", "leaf"):
            values = [output[f"d{depth}:{stage}:{level}"] for depth in DEPTHS]
            output[f"macro:{stage}:{level}"] = sum(values) / len(values)
    return output


def model_state_payload(model):
    state = d11.trainable_state(model)
    return state, c10.state_sha256(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--f16-cases", required=True)
    parser.add_argument("--eval-wmt-data", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--eval-per-class", type=int, default=8)
    parser.add_argument("--head-steps", type=int, default=50)
    parser.add_argument("--joint-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--wmt-eval-rows", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=2e-3)
    parser.add_argument("--joint-head-lr", type=float, default=5e-4)
    parser.add_argument("--model-lr", type=float, default=2e-5)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12901)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "trace.jsonl"

    payload, runtime_args, sp, model, pieces, eos, bos, source_hash = load_runtime(
        Path(args.checkpoint), args.device,
    )
    initial_source_hash = c10.state_sha256(model.frozen_source.state_dict())
    initial_state, initial_model_hash = model_state_payload(model)
    f16_payload = json.loads(Path(args.f16_cases).read_text(encoding="utf-8"))
    excluded_lines = {int(row["line"]) for row in f16_payload["rows"]}
    candidates, inventory = collect_candidates(
        Path(args.corpus), excluded_lines, args.candidate_cap, args.seed,
    )
    rows = build_manifest(candidates, sp, args.train_per_class, args.eval_per_class)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]

    serializable = [{key: value for key, value in row.items() if key not in {"source_ids", "target_ids"}} for row in rows]
    manifest_text = json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    write_json(output / "manifest.json", {
        "sha256": manifest_hash, "excluded_f16_lines": sorted(excluded_lines),
        "inventory": {"|".join(map(str, key)): value for key, value in inventory.items()},
        "rows": serializable,
    })

    f16_rows = []
    for row in f16_payload["rows"]:
        token_ids = sp.encode(row["zh"], out_type=int)
        f16_rows.append({**row, "token_ids": token_ids})

    runtime_args.eval_batch = args.eval_batch
    runtime_args.device = args.device
    direction_ids = {"en2zh": pieces + 1, "zh2en": pieces + 2}
    wmt_valid, _, _ = d10.collect_wmt_eval(
        Path(args.eval_wmt_data), sp, direction_ids, eos, args.wmt_eval_rows,
    )
    baseline_wmt = d10.valid_summary(model, wmt_valid, runtime_args, pieces, bos)
    baseline_f16 = f16_probe(model, f16_rows, pieces, eos, args.eval_batch, args.device)

    sample_source, sample_lengths, sample_target, _ = collate(
        train_rows[:args.batch_size], pieces, eos, args.device,
    )
    with torch.no_grad():
        _, sample_root, _, _, _, _, _ = forward_with_features(
            model, sample_source, sample_lengths, sample_target, bos, DEPTHS[0],
        )
    head = nn.Linear(sample_root.shape[-1], len(CONCEPT_NAMES)).to(args.device)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(head.bias)
    train_labels = torch.tensor([row["labels"] for row in train_rows], dtype=torch.float32)
    positives = train_labels.sum(0)
    pos_weight = ((len(train_rows) - positives) / positives.clamp_min(1.0)).to(args.device)

    head_optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr)
    finite = True
    for step, batch in enumerate(batches(train_rows, args.batch_size, args.head_steps, args.seed + 1), 1):
        model.eval()
        head.train()
        source, lengths, target, labels = collate(batch, pieces, eos, args.device)
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        with torch.no_grad():
            _, root, leaf, _, _, _, _ = forward_with_features(model, source, lengths, target, bos, depth)
        root_logits, leaf_logits = head(root), head(leaf)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(root_logits, labels, pos_weight=pos_weight)
            + F.binary_cross_entropy_with_logits(leaf_logits, labels, pos_weight=pos_weight)
        )
        head_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        finite = finite and math.isfinite(float(loss.detach())) and math.isfinite(grad_norm)
        if not finite:
            raise RuntimeError(f"non-finite head warmup at step {step}")
        head_optimizer.step()
        if step == 1 or step % 10 == 0:
            event = {"stage": "head", "step": step, "depth": depth, "aux_loss": float(loss.detach()), "grad_norm": grad_norm}
            append_jsonl(trace_path, event)
            print(json.dumps(event), flush=True)

    warm_head_eval = head_evaluation(
        model, head, eval_rows, pieces, eos, bos, args.eval_batch, args.device,
    )
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    joint_optimizer = torch.optim.AdamW([
        {"params": model_parameters, "lr": args.model_lr},
        {"params": list(head.parameters()), "lr": args.joint_head_lr},
    ])
    for step, batch in enumerate(batches(train_rows, args.batch_size, args.joint_steps, args.seed + 2), 1):
        model.train()
        head.train()
        source, lengths, target, labels = collate(batch, pieces, eos, args.device)
        depth = DEPTHS[(step + args.seed) % len(DEPTHS)]
        logits, root, leaf, _, _, slots, _ = forward_with_features(
            model, source, lengths, target, bos, depth,
        )
        tokens = int(target.ne(pieces).sum())
        translation_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=pieces, reduction="sum",
        ) / max(1, tokens)
        auxiliary_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(head(root), labels, pos_weight=pos_weight)
            + F.binary_cross_entropy_with_logits(head(leaf), labels, pos_weight=pos_weight)
        )
        loss = translation_loss + args.aux_weight * auxiliary_loss
        joint_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = finite and d07.finite_trainable_gradients(model)
        parameters = [*model_parameters, *head.parameters()]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        finite = finite and all(math.isfinite(value) for value in (
            float(loss.detach()), float(translation_loss.detach()), float(auxiliary_loss.detach()), grad_norm,
        ))
        if not finite:
            raise RuntimeError(f"non-finite joint state at step {step}")
        joint_optimizer.step()
        if step == 1 or step % 10 == 0:
            event = {
                "stage": "joint", "step": step, "depth": depth,
                "loss": float(loss.detach()), "translation_loss": float(translation_loss.detach()),
                "auxiliary_loss": float(auxiliary_loss.detach()), "grad_norm": grad_norm,
                "slot_variance": float(slots.detach().var()),
            }
            append_jsonl(trace_path, event)
            print(json.dumps(event), flush=True)

    model.eval()
    final_head_eval = head_evaluation(
        model, head, eval_rows, pieces, eos, bos, args.eval_batch, args.device,
    )
    final_wmt = d10.valid_summary(model, wmt_valid, runtime_args, pieces, bos)
    final_f16 = f16_probe(model, f16_rows, pieces, eos, args.eval_batch, args.device)
    final_state, final_model_hash = model_state_payload(model)
    final_source_hash = c10.state_sha256(model.frozen_source.state_dict())

    checkpoint = {
        "claim": CLAIM, "parent_checkpoint": args.checkpoint,
        "parent_trainable_state_sha256": payload["trainable_state_sha256"],
        "trainable_state_dict": final_state, "trainable_state_sha256": final_model_hash,
        "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "head_width": sample_root.shape[-1], "concepts": list(CONCEPT_NAMES),
        "manifest_sha256": manifest_hash, "config": vars(args),
    }
    checkpoint_path = output / "checkpoint_latest.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)

    reload_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_ok = (
        reload_payload["trainable_state_sha256"] == final_model_hash
        and c10.state_sha256(reload_payload["trainable_state_dict"]) == final_model_hash
    )
    head_rows = []
    for stage, values in (("head_warmup", warm_head_eval), ("joint_final", final_head_eval)):
        for row in values:
            head_rows.append({"stage": stage, **row})
    with (output / "head_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(head_rows[0]))
        writer.writeheader()
        writer.writerows(head_rows)

    def head_macro(rows, level):
        values = [row["auroc"] for row in rows if row["level"] == level]
        return sum(values) / len(values)

    final_head_root = head_macro(final_head_eval, "root")
    final_head_leaf = head_macro(final_head_eval, "leaf")
    wmt_delta = final_wmt["mean_nll"] - baseline_wmt["mean_nll"]
    fold_root_delta = final_f16["macro:fold:root"] - baseline_f16["macro:fold:root"]
    read_root_delta = final_f16["macro:read_facing:root"] - baseline_f16["macro:read_facing:root"]
    expected_family_count = sum(len(spec["families"]) for spec in CONCEPTS.values())
    split_by_line = collections.defaultdict(set)
    for row in rows:
        split_by_line[row["line"]].add(row["split"])
    gates = {
        "O0_data_contract": (
            expected_family_count == 23
            and len(train_rows) == 23 * args.train_per_class * 2
            and len(eval_rows) == 23 * args.eval_per_class * 2
            and not ({row["line"] for row in rows} & excluded_lines)
            and all(len(splits) == 1 for splits in split_by_line.values())
        ),
        "O1_finite": finite,
        "O2_source_frozen": initial_source_hash == final_source_hash == source_hash,
        "O2_model_updated": initial_model_hash != final_model_hash,
        "O3_checkpoint_reload": reload_ok,
        "S0_wmt_nll_preserved": wmt_delta <= 0.20,
        "P1_aux_root_auroc": final_head_root >= 0.65,
        "P2_aux_scale_consistency": abs(final_head_root - final_head_leaf) <= 0.05,
        "P3_f16_fold_root_transfer": fold_root_delta >= 0.03,
        "P4_f16_read_root_transfer": read_root_delta >= 0.03,
    }
    if gates["P1_aux_root_auroc"] and gates["P3_f16_fold_root_transfer"] and gates["P4_f16_read_root_transfer"]:
        diagnosis = "auxiliary semantics transfer to excluded push families"
    elif gates["P1_aux_root_auroc"]:
        diagnosis = "auxiliary task learned in-distribution directions without excluded-family transfer"
    elif final_head_leaf >= 0.65 and final_head_root < final_head_leaf - 0.05:
        diagnosis = "auxiliary direction forms at leaf but degrades at root; revisit FOLD"
    else:
        diagnosis = "semantic pressure insufficient at both scales; do not isolate FOLD"

    summary = {
        "claim": CLAIM, "host": os.uname().nodename,
        "checkpoint": args.checkpoint, "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_state_sha256": source_hash, "manifest_sha256": manifest_hash,
        "train_instances": len(train_rows), "eval_instances": len(eval_rows),
        "unique_corpus_lines": len(split_by_line), "concepts": list(CONCEPT_NAMES),
        "family_count": expected_family_count, "head_width": sample_root.shape[-1],
        "head_macro_auroc": {
            "warmup_root": head_macro(warm_head_eval, "root"),
            "warmup_leaf": head_macro(warm_head_eval, "leaf"),
            "final_root": final_head_root, "final_leaf": final_head_leaf,
        },
        "wmt": {"before": baseline_wmt, "after": final_wmt, "mean_nll_delta": wmt_delta},
        "f16_before": baseline_f16, "f16_after": final_f16,
        "f16_macro_delta": {"fold_root": fold_root_delta, "read_facing_root": read_root_delta},
        "initial_model_sha256": initial_model_hash, "final_model_sha256": final_model_hash,
        "checkpoint_file": str(checkpoint_path), "gates": gates,
        "diagnosis": diagnosis, "seconds": time.time() - started,
        "claim_boundary": "small automatic-label task training; not product translation proof",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        f"# {CLAIM}\n\n- Train/eval instances: `{len(train_rows)}/{len(eval_rows)}`\n"
        f"- Head AUROC: `{summary['head_macro_auroc']}`\n- WMT NLL delta: `{wmt_delta}`\n"
        f"- F16 root deltas: `{summary['f16_macro_delta']}`\n- Gates: `{gates}`\n"
        f"- Diagnosis: `{diagnosis}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
