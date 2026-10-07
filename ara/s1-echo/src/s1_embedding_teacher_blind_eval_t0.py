#!/usr/bin/env python3
"""Blind local-teacher audit for T0 embedding nearest neighbours.

Owner: Nio / Trinity
Author: GPT-5.6 Sol
Created: 2026-10-07
Updated: 2026-10-07
Responsibility: calibrate a frozen local teacher on SimLex rankings, then judge
order-swapped nearest-neighbour comparisons without feeding labels into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def calibration_items(rows: list[dict], limit: int, seed: int) -> list[dict]:
    by_anchor: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        left, right, score = str(row["word1"]), str(row["word2"]), float(row["human"])
        by_anchor[left].append((right, score))
        by_anchor[right].append((left, score))
    candidates = []
    for anchor, values in by_anchor.items():
        ordered = sorted(values, key=lambda item: item[1])
        if len(ordered) < 2 or ordered[-1][1] - ordered[0][1] < 2.0:
            continue
        candidates.append({
            "kind": "calibration",
            "anchor": anchor,
            "option_a": ordered[-1][0],
            "option_b": ordered[0][0],
            "gold": "A",
            "score_a": ordered[-1][1],
            "score_b": ordered[0][1],
        })
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def prompt(item: dict) -> str:
    return f"""你是词义相似度盲评员。只判断哪个候选词与目标词的含义更相似，不按拼写、词频或共同出现次数判断。

目标词：{item['anchor']}
候选 A：{item['option_a']}
候选 B：{item['option_b']}

只能输出一个 JSON 对象：
{{"choice":"A|B|TIE","reason_zh":"一句简短中文"}}
不要输出 Markdown，不要解释评分过程。"""


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("choice") not in {"A", "B", "TIE"}:
        return None
    return value


@torch.inference_mode()
def generate(model, tokenizer, text: str, max_input: int, max_output: int) -> str:
    chat = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    batch = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    input_ids = batch.input_ids[:, -max_input:].to("cuda")
    attention = batch.attention_mask[:, -max_input:].to("cuda")
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention,
        max_new_tokens=max_output,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def swap(item: dict) -> dict:
    result = dict(item)
    result["option_a"], result["option_b"] = item["option_b"], item["option_a"]
    if item.get("gold") == "A":
        result["gold"] = "B"
    elif item.get("gold") == "B":
        result["gold"] = "A"
    if "hidden_a" in item:
        result["hidden_a"], result["hidden_b"] = item["hidden_b"], item["hidden_a"]
    return result


def consistent(left: str, right: str) -> bool:
    return (left == "A" and right == "B") or (left == "B" and right == "A") or (left == right == "TIE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/home/nio/models/Qwen3-8B"))
    parser.add_argument("--embedding-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-items", type=int, default=40)
    parser.add_argument("--blind-items", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--max-output", type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows_path = args.embedding_evidence / "simlex_rows.json"
    blind_path = args.embedding_evidence / "teacher_items.jsonl"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    calibration = calibration_items(rows, args.calibration_items, args.seed)
    blind = read_jsonl(blind_path)[: args.blind_items]
    items = calibration + blind
    if not calibration or not blind:
        raise RuntimeError(f"insufficient teacher material: calibration={len(calibration)} blind={len(blind)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    outputs = []
    for index, item in enumerate(items):
        forward_raw = generate(model, tokenizer, prompt(item), args.max_input, args.max_output)
        swapped_item = swap(item)
        swapped_raw = generate(model, tokenizer, prompt(swapped_item), args.max_input, args.max_output)
        forward = parse_json(forward_raw)
        backward = parse_json(swapped_raw)
        outputs.append({
            "index": index,
            "item": item,
            "forward_raw": forward_raw,
            "swapped_raw": swapped_raw,
            "forward": forward,
            "swapped": backward,
            "order_consistent": bool(forward and backward and consistent(forward["choice"], backward["choice"])),
        })
        print(json.dumps({"done": index + 1, "total": len(items), "kind": item["kind"], "parsed": bool(forward and backward)}), flush=True)

    parsed = [row for row in outputs if row["forward"] and row["swapped"]]
    calibration_rows = [row for row in parsed if row["item"]["kind"] == "calibration"]
    blind_rows = [row for row in parsed if row["item"]["kind"] == "blind_neighbor"]
    calibration_accuracy = sum(row["forward"]["choice"] == row["item"]["gold"] for row in calibration_rows) / max(1, len(calibration_rows))
    order_consistency = sum(row["order_consistent"] for row in parsed) / max(1, len(parsed))
    comparisons: dict[str, dict[str, int]] = defaultdict(lambda: {"tree": 0, "control": 0, "tie": 0})
    for row in blind_rows:
        item = row["item"]
        choice = row["forward"]["choice"]
        selected = None if choice == "TIE" else item[f"hidden_{choice.lower()}"]
        control = item["hidden_b"] if item["hidden_a"] == "tree" else item["hidden_a"]
        if selected == "tree":
            comparisons[control]["tree"] += 1
        elif selected == control:
            comparisons[control]["control"] += 1
        else:
            comparisons[control]["tie"] += 1

    summary = {
        "schema": "nio.teacher-blind-eval.t0.v1",
        "model": str(args.model),
        "model_config_sha256": sha256(args.model / "config.json"),
        "embedding_summary_sha256": sha256(args.embedding_evidence / "summary.json"),
        "items": len(items),
        "calibration_items": len(calibration),
        "blind_items": len(blind),
        "paired_parse_rate": len(parsed) / len(items),
        "calibration_accuracy": calibration_accuracy,
        "order_consistency": order_consistency,
        "blind_comparisons": dict(comparisons),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.time() - started,
        "gates": {
            "parse_rate_ge_0_95": len(parsed) / len(items) >= 0.95,
            "calibration_accuracy_ge_0_75": calibration_accuracy >= 0.75,
            "order_consistency_ge_0_90": order_consistency >= 0.90,
        },
        "boundary": "Supplemental local-teacher observation only; no labels enter embedding construction.",
    }
    with (args.output / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

