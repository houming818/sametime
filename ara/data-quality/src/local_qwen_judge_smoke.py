#!/usr/bin/env python3
"""Deterministic local-Qwen structured-judge smoke over scored review samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


FAMILIES = ("mono", "qa", "medical")
PICKS = (0, 12, 25, 37, 50, 62, 75, 87, 100, 110, 120, 130, 140, 149, 150, 159, 169, 179, 189, 199)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> list[dict]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result.append({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_records(evidence_root: Path) -> tuple[list[dict], dict[str, str]]:
    records = []
    source_hashes = {}
    for family in FAMILIES:
        path = evidence_root / f"{family}_smoke_seed15101" / "review.jsonl"
        source_hashes[str(path)] = file_sha256(path)
        review = read_jsonl(path)
        if len(review) < 200:
            raise RuntimeError(f"expected at least 200 review records for {family}, got {len(review)}")
        for index in PICKS:
            record = dict(review[index])
            record["family"] = family
            record["review_index"] = index
            records.append(record)
    return records, source_hashes


def prompt_for(record: dict) -> str:
    medical = record["family"] == "medical"
    left = record["left"].replace("<|im_start|>", "< |im_start|>").replace("<|im_end|>", "< |im_end|>")
    right = record["right"].replace("<|im_start|>", "< |im_start|>").replace("<|im_end|>", "< |im_end|>")
    return f"""你是训练语料校准审核器，只评价两段文本之间的关系和文本质量，不改写原文，不认证事实正确性。

数据类型：{record['family']}
第一段：
<LEFT>
{left}
</LEFT>
第二段：
<RIGHT>
{right}
</RIGHT>

relation 只能是 matched、partial、mismatch、uncertain。
text_quality 只能是 usable、noisy、corrupt、uncertain。
domain_risk 在医疗数据中必须是 medical_unverified，其他数据必须是 ordinary。
reason_code 使用简短小写英文和下划线；reason_zh 使用一句简短中文。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要思考过程。医疗数据={str(medical).lower()}。
"""


def parse_exact_json(text: str) -> dict | None:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_input: int, max_output: int) -> str:
    # This is the checkpoint's checked-in chat template with thinking disabled.
    # Rendering it directly avoids depending on the host's unrelated Jinja version.
    chat = (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    batch = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    encoded = batch.input_ids[:, -max_input:].to("cuda")
    attention_mask = batch.attention_mask[:, -max_input:].to("cuda")
    output = model.generate(
        input_ids=encoded,
        attention_mask=attention_mask,
        max_new_tokens=max_output,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, encoded.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input", type=int, default=4096)
    parser.add_argument("--max-output", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    records, source_hashes_before = sample_records(args.evidence_root)
    model_manifest = tree_manifest(args.model)
    (args.output / "model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()

    outputs = []
    for position, record in enumerate(records):
        raw = generate_one(model, tokenizer, prompt_for(record), args.max_input, args.max_output)
        parsed = parse_exact_json(raw)
        outputs.append({
            "position": position,
            "family": record["family"],
            "source": record["source"],
            "row_id": record["row_id"],
            "review_index": record["review_index"],
            "bge_score": record["score"],
            "left": record["left"],
            "right": record["right"],
            "raw_output": raw,
            "parsed": parsed,
        })
        print(json.dumps({"done": position + 1, "family": record["family"], "parsed": parsed is not None}), flush=True)

    repeat_outputs = []
    for record in records[:6]:
        repeat_outputs.append(generate_one(model, tokenizer, prompt_for(record), args.max_input, args.max_output))

    source_hashes_after = {
        path: file_sha256(Path(path)) for path in source_hashes_before
    }
    parsed_count = sum(item["parsed"] is not None for item in outputs)
    medical_items = [item for item in outputs if item["family"] == "medical"]
    medical_risk_ok = all(
        item["parsed"] is not None and item["parsed"].get("domain_risk") == "medical_unverified"
        for item in medical_items
    )
    deterministic = all(outputs[index]["raw_output"] == repeat_outputs[index] for index in range(6))
    summary = {
        "claim": "NIO-LOCAL-JUDGE-C01",
        "model": str(args.model),
        "rows": len(outputs),
        "families": {family: sum(item["family"] == family for item in outputs) for family in FAMILIES},
        "parsed_json": parsed_count,
        "parse_rate": parsed_count / len(outputs),
        "repeat_exact_6": deterministic,
        "medical_risk_all_unverified": medical_risk_ok,
        "source_hashes_unchanged": source_hashes_before == source_hashes_after,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "gates": {
            "P1_gpu_inference": True,
            "P2_json_57_of_60": parsed_count >= 57,
            "P3_repeat_exact": deterministic,
            "P4_medical_unverified": medical_risk_ok,
            "P5_source_unchanged": source_hashes_before == source_hashes_after,
        },
        "seconds": time.time() - started,
        "warning": "Calibration judge smoke only; no factual or medical correctness certification.",
    }
    with (args.output / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output / "repeat_outputs.json").write_text(
        json.dumps(repeat_outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
