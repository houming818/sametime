#!/usr/bin/env python3
"""Probe target-side push semantics across held-out Chinese expression families."""
from __future__ import annotations

import argparse
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

import s3_filter_guided_theta_calibration_f04 as f04
import s3_treeheap_root_signal_recoverability_f15 as f15
from treeheap_epoch_translate_cli import DEPTHS, load_runtime


CLAIM = "S3-TREEHEAP-CROSS-LANGUAGE-SEMANTIC-RECOVERABILITY-F16"
PUSH = re.compile(r"(?i)\bpush(?:es|ed|ing)?\b")
CJK = re.compile(r"[\u4e00-\u9fff]")
FAMILIES = (
    ("physical", re.compile(r"推着|推开|推入|推倒|推向|推过|推车|推石|推门|推球|推人")),
    ("technical", re.compile(r"推送|推流|推到.{0,8}(库|仓|服务器)|版本库")),
    ("abstract", re.compile(r"推动|推进|推广|推行|推出|推迟|推崇|推举")),
    ("press", re.compile(r"按|压")),
    ("force", re.compile(r"迫使|逼|驱使|强迫")),
    ("other_tui", re.compile(r"推")),
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_family(text: str) -> str | None:
    for name, pattern in FAMILIES:
        if pattern.search(text):
            return name
    return None


def reservoir_candidates(path: Path, cap: int, seed: int) -> dict[tuple[str, int], list[dict]]:
    buckets = {(name, label): [] for name, _ in FAMILIES for label in (0, 1)}
    seen = {key: 0 for key in buckets}
    rng = random.Random(seed)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 2:
                continue
            zh, en = columns
            family = source_family(zh)
            if family is None or len(CJK.findall(zh)) < 4 or "�" in zh or "�" in en:
                continue
            label = int(bool(PUSH.search(en)))
            key = (family, label)
            seen[key] += 1
            row = {"line": line_number, "zh": zh, "en": en, "family": family, "label": label}
            bucket = buckets[key]
            if len(bucket) < cap:
                bucket.append(row)
            else:
                replacement = rng.randrange(seen[key])
                if replacement < cap:
                    bucket[replacement] = row
    return buckets


def select_cases(buckets, sp, per_class: int) -> list[dict]:
    selected = []
    for family, _ in FAMILIES:
        by_label = {}
        for label in (0, 1):
            valid = []
            for row in buckets[(family, label)]:
                token_ids = sp.encode(row["zh"], out_type=int)
                if 2 <= len(token_ids) <= 32:
                    valid.append({**row, "token_ids": token_ids, "source_pieces": len(token_ids)})
            by_label[label] = valid
        positives = sorted(by_label[1], key=lambda row: (row["source_pieces"], row["line"]))
        negatives = by_label[0][:]
        if len(positives) < per_class or len(negatives) < per_class:
            raise ValueError(
                f"insufficient {family}: positive={len(positives)} negative={len(negatives)}"
            )
        # Greedy nearest-length pairing limits the easiest nuisance variable.
        pairs = []
        used = set()
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
            f"{family}:{pair[0]['line']}:{pair[1]['line']}".encode("utf-8")
        ).digest())
        for positive, negative in pairs[:per_class]:
            selected.extend((positive, negative))
    selected.sort(key=lambda row: (row["family"], row["label"], row["line"]))
    return selected


def collate(rows: list[dict], pieces: int, eos: int, device: str):
    width = 34
    source = torch.full((len(rows), width), pieces, dtype=torch.long, device=device)
    lengths = torch.empty(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        encoded = [pieces + 2, *row["token_ids"], eos]
        source[index, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
        lengths[index] = len(encoded)
    return source, lengths


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Average tied ranks; ties are uncommon but input BOW can create them.
    values, inverse, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    if bool((counts > 1).any()):
        for group in range(len(values)):
            members = inverse == group
            ranks[members] = ranks[members].mean()
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    ranked = labels[order].double()
    precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1, dtype=torch.float64)
    return float((precision * ranked).sum() / ranked.sum().clamp_min(1.0))


def fit_direction(features, labels, train_mask, test_mask, shuffled_labels=None) -> dict:
    x_train = features[train_mask].double()
    x_test = features[test_mask].double()
    y_train = (shuffled_labels if shuffled_labels is not None else labels[train_mask]).long()
    y_test = labels[test_mask].long()
    mean = x_train.mean(0)
    scale = x_train.std(0, unbiased=False)
    active = scale > 1e-8
    train = (x_train[:, active] - mean[active]) / scale[active]
    test = (x_test[:, active] - mean[active]) / scale[active]
    direction = train[y_train == 1].mean(0) - train[y_train == 0].mean(0)
    train_scores = train @ direction
    scores = test @ direction
    threshold = 0.5 * (train_scores[y_train == 1].mean() + train_scores[y_train == 0].mean())
    prediction = scores >= threshold
    positive = y_test == 1
    negative = ~positive
    result = {
        "auroc": auc(scores, y_test),
        "average_precision": average_precision(scores, y_test),
        "balanced_accuracy": float(0.5 * (
            prediction[positive].double().mean() + (~prediction[negative]).double().mean()
        )),
        "score_gap": float(scores[positive].mean() - scores[negative].mean()),
        "active_features": int(active.sum()),
        "finite": bool(torch.isfinite(scores).all()),
    }
    return result


def shuffled_train_labels(rows, train_indices, seed: int) -> torch.Tensor:
    output = torch.tensor([rows[index]["label"] for index in train_indices], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    families = sorted({rows[index]["family"] for index in train_indices})
    for family in families:
        positions = [offset for offset, index in enumerate(train_indices) if rows[index]["family"] == family]
        values = output[positions]
        permutation = torch.randperm(len(positions), generator=generator)
        output[positions] = values[permutation]
    return output


def extract_features(model, rows, pieces, eos, batch_size, device):
    features = {depth: {"fold": {}, "read_facing": {}} for depth in DEPTHS}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        source, lengths = collate(batch, pieces, eos, device)
        for depth in DEPTHS:
            with torch.no_grad():
                protocol = model.protocol(source, lengths, depth)
                stages = f15.extract_level_features(model, protocol)
            for stage, levels in stages.items():
                chosen = {"root": 0, "middle": len(levels) // 2, "leaf": len(levels) - 1}
                for label, level in chosen.items():
                    features[depth][stage].setdefault(label, []).append(levels[level])
    return {
        depth: {
            stage: {label: torch.cat(parts, 0) for label, parts in levels.items()}
            for stage, levels in stages.items()
        }
        for depth, stages in features.items()
    }


def input_bow(rows, pieces: int) -> torch.Tensor:
    result = torch.zeros((len(rows), pieces), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for token in row["token_ids"]:
            result[row_index, token] += 1.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12701)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    started = time.time()
    output = Path(args.evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload, _, sp, model, pieces, eos, _, source_hash = load_runtime(
        Path(args.checkpoint), args.device,
    )
    f04.freeze_model(model)
    expected_state = payload["trainable_state_dict"]

    buckets = reservoir_candidates(Path(args.corpus), args.candidate_cap, args.seed)
    rows = select_cases(buckets, sp, args.per_class)
    manifest = [{key: value for key, value in row.items() if key != "token_ids"} for row in rows]
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    write_json(output / "cases.json", {"sha256": manifest_hash, "rows": manifest})

    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    model_features = extract_features(model, rows, pieces, eos, args.batch_size, args.device)
    all_features = {"input_bow": input_bow(rows, pieces)}
    for depth, stages in model_features.items():
        for stage, levels in stages.items():
            for level, values in levels.items():
                all_features[f"d{depth}:{stage}:{level}"] = values

    result_rows = []
    finite = True
    family_names = [name for name, _ in FAMILIES]
    for held_out in family_names:
        train_indices = [index for index, row in enumerate(rows) if row["family"] != held_out]
        test_indices = [index for index, row in enumerate(rows) if row["family"] == held_out]
        train_mask = torch.zeros(len(rows), dtype=torch.bool)
        test_mask = torch.zeros(len(rows), dtype=torch.bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        shuffled = [
            shuffled_train_labels(rows, train_indices, args.seed + 1000 * repeat + len(held_out))
            for repeat in range(args.shuffle_repeats)
        ]
        for feature_name, values in all_features.items():
            metrics = fit_direction(values, labels, train_mask, test_mask)
            controls = [
                fit_direction(values, labels, train_mask, test_mask, control)["auroc"]
                for control in shuffled
            ]
            finite = finite and metrics["finite"] and all(math.isfinite(value) for value in controls)
            result_rows.append({
                "held_out_family": held_out,
                "feature": feature_name,
                **metrics,
                "shuffled_auroc_mean": sum(controls) / len(controls),
                "shuffled_auroc_max": max(controls),
            })
    write_csv(output / "probe_results.csv", result_rows)

    def macro(feature_suffix: str, field: str = "auroc") -> float:
        values = [row[field] for row in result_rows if row["feature"].endswith(feature_suffix)]
        if not values:
            raise ValueError(f"missing feature suffix {feature_suffix}")
        return sum(values) / len(values)

    fold_root = macro(":fold:root")
    fold_leaf = macro(":fold:leaf")
    read_root = macro(":read_facing:root")
    read_leaf = macro(":read_facing:leaf")
    read_root_controls = [
        row["shuffled_auroc_mean"] for row in result_rows
        if row["feature"].endswith(":read_facing:root")
    ]
    counts = {
        family: {
            str(label): sum(row["family"] == family and row["label"] == label for row in rows)
            for label in (0, 1)
        }
        for family in family_names
    }
    gates = {
        "O0_balanced_complete_family_holdout": (
            len(rows) == len(FAMILIES) * args.per_class * 2
            and all(value == args.per_class for family in counts.values() for value in family.values())
            and len({row["line"] for row in rows}) == len(rows)
        ),
        "O1_finite": finite,
        "O2_frozen_checkpoint": f04.frozen_exact(model, expected_state),
        "C0_shuffled_control_rejected": sum(read_root_controls) / len(read_root_controls) < 0.60,
        "P1_fold_root_semantic_recoverable": fold_root >= 0.65,
        "P2_read_root_semantic_recoverable": read_root >= 0.65,
        "P3_fold_compression_loss_pattern": fold_leaf >= 0.70 and fold_root <= fold_leaf - 0.10,
        "P3_read_compression_loss_pattern": read_leaf >= 0.70 and read_root <= read_leaf - 0.10,
    }
    if gates["P3_fold_compression_loss_pattern"]:
        diagnosis = "supports target-label information loss across FOLD compression"
    elif gates["P3_read_compression_loss_pattern"]:
        diagnosis = "FOLD retains more target-label information than READ-facing root"
    elif gates["P1_fold_root_semantic_recoverable"] and gates["P2_read_root_semantic_recoverable"]:
        diagnosis = "target-label direction remains recoverable at root; prioritize decoder alignment"
    elif fold_leaf < 0.70 and read_leaf < 0.70:
        diagnosis = "no strong target-label direction even at leaf; compression is not isolated"
    else:
        diagnosis = "mixed result; inspect held-out families before structural change"

    summary = {
        "claim": CLAIM,
        "host": os.uname().nodename,
        "checkpoint": args.checkpoint,
        "checkpoint_state_sha256": payload["trainable_state_sha256"],
        "source_state_sha256": source_hash,
        "corpus": args.corpus,
        "sample_manifest_sha256": manifest_hash,
        "case_count": len(rows),
        "family_counts": counts,
        "protocol_depths": list(DEPTHS),
        "macro_auroc": {
            "input_bow": macro("input_bow"),
            "fold_root": fold_root,
            "fold_leaf": fold_leaf,
            "read_facing_root": read_root,
            "read_facing_leaf": read_leaf,
        },
        "gates": gates,
        "diagnosis": diagnosis,
        "seconds": time.time() - started,
        "claim_boundary": "automatic target push-presence prediction, not complete translation semantics",
    }
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        f"# {CLAIM}\n\n- Cases: `{len(rows)}`\n- Macro AUROC: `{summary['macro_auroc']}`\n"
        f"- Gates: `{gates}`\n- Diagnosis: `{diagnosis}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
