#!/usr/bin/env python3
"""Blind-review bilingual corpus pairs with the DeepSeek API."""

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


LABELS = {"correct", "partial", "mismatch", "uncertain"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def call_api(api_key: str, model: str, rows: list[dict], retries: int = 5) -> list[dict]:
    items = [{"review_id": r["review_id"], "zh": r["zh"], "en": r["en"]} for r in rows]
    prompt = {
        "task": "Judge whether each Chinese-English pair expresses the same meaning.",
        "labels": {
            "correct": "Faithful translation; minor style differences are acceptable.",
            "partial": "Core meaning overlaps, but important information is missing or added.",
            "mismatch": "Different meanings or unrelated texts.",
            "uncertain": "Insufficient context, corrupted text, or genuinely hard to judge.",
        },
        "rules": [
            "Judge semantics, not literal word overlap.",
            "Be strict about negation, numbers, names, and direction of events.",
            "Return every review_id exactly once.",
            "confidence must be a number from 0 to 1.",
            "reason must be concise Chinese.",
        ],
        "input": items,
        "output_json_example": {
            "reviews": [
                {"review_id": "DQ-0001", "label": "mismatch", "confidence": 0.99, "reason": "中英文主题无关"}
            ]
        },
    }
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bilingual corpus auditor. Output valid JSON only.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 2400,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            reviews = json.loads(content)["reviews"]
            expected = {r["review_id"] for r in rows}
            actual = {r.get("review_id") for r in reviews}
            if expected != actual:
                raise ValueError(f"review_id mismatch: missing={expected-actual}, extra={actual-expected}")
            for review in reviews:
                if review.get("label") not in LABELS:
                    raise ValueError(f"invalid label: {review.get('label')}")
                review["confidence"] = max(0.0, min(1.0, float(review.get("confidence", 0))))
            return reviews
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    load_env(args.env)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing")
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["review_id"]] = item

    pending = [row for row in rows if row["review_id"] not in completed]
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            reviews = call_api(api_key, args.model, batch)
            for review in reviews:
                handle.write(json.dumps(review, ensure_ascii=False) + "\n")
                completed[review["review_id"]] = review
            handle.flush()
            print(f"reviewed={len(completed)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
