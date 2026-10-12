#!/usr/bin/env python3
"""Generate auditable multiresolution Chinese targets with a frozen local teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Iterable


FIELDS = ("core", "short", "reduced")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_length(text: str) -> int:
    return len(re.sub(r"[\s，。！？；：、,.!?;:'\"“”‘’()（）\[\]【】<>《》-]", "", text))


def eligible(zh: str, en: str) -> bool:
    zh_len = compact_length(zh)
    en_len = len(en.split())
    return (
        28 <= zh_len <= 140
        and 8 <= en_len <= 80
        and "http://" not in zh
        and "https://" not in zh
        and "http://" not in en
        and "https://" not in en
        and not zh.isascii()
    )


def select_material(path: Path, count: int, seed: int) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    selected: list[dict] = []
    eligible_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            zh, en = (part.strip() for part in parts)
            if not eligible(zh, en):
                continue
            eligible_count += 1
            row = {"source_line": line_number, "source_en": en, "target_zh_full": zh}
            if len(selected) < count:
                selected.append(row)
            else:
                replacement = rng.randrange(eligible_count)
                if replacement < count:
                    selected[replacement] = row
    if len(selected) < count:
        raise RuntimeError(f"insufficient eligible rows: requested={count} selected={len(selected)}")
    selected.sort(key=lambda row: row["source_line"])
    return selected, eligible_count


def json_object(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    result = {}
    for field in FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            return None
        result[field] = item.strip()
    return result


def length_ranges(full: str) -> dict[str, tuple[int, int]]:
    length = compact_length(full)
    return {
        "core": (2, min(24, max(6, round(length * 0.30)))),
        "short": (
            max(4, round(length * 0.18)),
            min(48, max(8, round(length * 0.50))),
        ),
        "reduced": (
            max(8, round(length * 0.40)),
            max(16, round(length * 0.78)),
        ),
    }


def validate_candidate(candidate: dict | None, full: str) -> tuple[bool, list[str], dict]:
    if candidate is None:
        return False, ["json_parse_or_schema"], {}
    lengths = {field: compact_length(candidate[field]) for field in FIELDS}
    lengths["full"] = compact_length(full)
    ranges = length_ranges(full)
    reasons = []
    if not (0 < lengths["core"] < lengths["short"] < lengths["reduced"] < lengths["full"]):
        reasons.append("non_monotonic_length")
    for field in FIELDS:
        minimum, maximum = ranges[field]
        if lengths[field] < minimum:
            reasons.append(f"{field}_under_min")
        if lengths[field] > maximum:
            reasons.append(f"{field}_over_max")
    if candidate["short"] == candidate["reduced"]:
        reasons.append("duplicate_short_reduced")
    if candidate["reduced"] == full.strip():
        reasons.append("reduced_equals_full")
    ratios = {field: lengths[field] / max(1, lengths["full"]) for field in FIELDS}
    return not reasons, reasons, {"lengths": lengths, "ratios": ratios, "ranges": ranges}


def prompt(row: dict) -> str:
    ranges = length_ranges(row["target_zh_full"])
    return f"""你是多分辨率中文语料工程师。给定一组已经对齐的英文源句和中文完整译文，只压缩中文译文，不重新翻译英文。

英文源句：{row['source_en']}
中文完整译文：{row['target_zh_full']}

请生成三个逐级压缩的中文目标：
1. core：核心词组，保留主体、动作、关键对象，不要求是完整句，长度 {ranges['core'][0]} 到 {ranges['core'][1]} 个汉字或字母数字字符；
2. short：可独立理解的极短句，长度 {ranges['short'][0]} 到 {ranges['short'][1]} 个汉字或字母数字字符；
3. reduced：语法完整的缩句，长度 {ranges['reduced'][0]} 到 {ranges['reduced'][1]} 个汉字或字母数字字符。

约束：
- 长度必须严格满足 core < short < reduced < 中文完整译文；
- short 必须比 core 至少多表达一项关系，reduced 必须比 short 至少多保留一项事实；
- 不得引入原文没有的事实；
- 必须保留否定、数量、主体变化和关键因果；
- reduced 必须比完整译文短；
- 优先抽取原译文已有词语，避免固定套话。

只能输出一个 JSON 对象，不要 Markdown，不要分析：
{{"core":"...","short":"...","reduced":"..."}}"""


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/nio/models/Qwen3-8B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--max-output", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    material, eligible_count = select_material(args.data, args.items, args.seed)
    material_path = args.output / "input_material.jsonl"
    append_jsonl(material_path, material)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    records = []
    accepted = []
    parse_count = 0
    monotonic_count = 0
    for index, row in enumerate(material):
        text = prompt(row)
        chat = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        batch = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
        input_ids = batch.input_ids[:, -args.max_input :].to("cuda")
        attention_mask = batch.attention_mask[:, -args.max_input :].to("cuda")
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_output,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
        candidate = json_object(raw)
        if candidate is not None:
            parse_count += 1
        valid, reasons, audit = validate_candidate(candidate, row["target_zh_full"])
        if candidate is not None:
            lens = audit.get("lengths", {})
            if lens and 0 < lens["core"] <= lens["short"] <= lens["reduced"] < lens["full"]:
                monotonic_count += 1
        record = {
            "index": index,
            **row,
            "teacher_synthetic": candidate,
            "raw_teacher_output": raw,
            "accepted": valid,
            "rejection_reasons": reasons,
            "audit": audit,
        }
        records.append(record)
        if valid and candidate is not None:
            accepted.append({
                "source_line": row["source_line"],
                "source_en": row["source_en"],
                "target_zh_core": candidate["core"],
                "target_zh_short": candidate["short"],
                "target_zh_reduced": candidate["reduced"],
                "target_zh_full": row["target_zh_full"],
                "provenance": "teacher_synthetic_qwen3_8b_t0",
            })
        print(json.dumps({"done": index + 1, "total": len(material), "accepted": len(accepted)}, ensure_ascii=False), flush=True)

    all_path = args.output / "teacher_candidates.jsonl"
    accepted_path = args.output / "accepted_curriculum.jsonl"
    append_jsonl(all_path, records)
    append_jsonl(accepted_path, accepted)
    total = len(material)
    summary = {
        "schema": "nio.s3.multiresolution-language-curriculum.t0.v1",
        "claim": "S3-MULTIRES-CURRICULUM-T0-R1-C01",
        "config": {"items": args.items, "seed": args.seed, "max_input": args.max_input, "max_output": args.max_output},
        "inputs": {
            "data": str(args.data),
            "data_sha256": sha256(args.data),
            "eligible_rows": eligible_count,
            "material_sha256": sha256(material_path),
        },
        "teacher": {
            "model": str(args.model),
            "config_sha256": sha256(args.model / "config.json"),
            "decode": "greedy",
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
        "results": {
            "rows": total,
            "parsed": parse_count,
            "monotonic": monotonic_count,
            "accepted": len(accepted),
            "parse_rate": parse_count / max(1, total),
            "monotonic_rate": monotonic_count / max(1, total),
            "acceptance_rate": len(accepted) / max(1, total),
        },
        "gates": {
            "parse_rate_ge_0_95": parse_count / max(1, total) >= 0.95,
            "monotonic_rate_ge_0_85": monotonic_count / max(1, total) >= 0.85,
            "acceptance_rate_ge_0_80": len(accepted) / max(1, total) >= 0.80,
        },
        "artifacts": {
            "all_sha256": sha256(all_path),
            "accepted_sha256": sha256(accepted_path),
        },
        "seconds": time.time() - started,
        "boundary": "Teacher-synthetic candidates only; formal gates do not establish semantic faithfulness or downstream TreeHeap benefit.",
    }
    summary["formal_release_allowed"] = all(summary["gates"].values())
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

