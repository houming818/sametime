#!/usr/bin/env python3
"""Audit a D10 recovery checkpoint without mutating it."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


CLAIM = "S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_audit(value, path="root"):
    count = 0
    nonfinite = []
    if torch.is_tensor(value):
        count = 1
        if value.is_floating_point() and not torch.isfinite(value).all():
            nonfinite.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            child_count, child_bad = tensor_audit(item, f"{path}.{key}")
            count += child_count
            nonfinite.extend(child_bad)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_count, child_bad = tensor_audit(item, f"{path}[{index}]")
            count += child_count
            nonfinite.extend(child_bad)
    return count, nonfinite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-cursor", type=int, default=-1)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors, nonfinite = tensor_audit(payload)
    gates = {
        "claim": payload.get("claim") == CLAIM,
        "stage": payload.get("stage") == "task",
        "step": int(payload.get("step", -1)) == args.expected_step,
        "cursor": args.expected_cursor < 0
        or int(payload.get("cursor", -1)) == args.expected_cursor,
        "optimizer": isinstance(payload.get("optimizer_state_dict"), dict),
        "trainable_state": isinstance(payload.get("trainable_state_dict"), dict),
        "finite": not nonfinite,
    }
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "sha256": file_sha256(checkpoint),
        "claim": payload.get("claim"),
        "stage": payload.get("stage"),
        "step": payload.get("step"),
        "cursor": payload.get("cursor"),
        "trainable_state_sha256": payload.get("trainable_state_sha256"),
        "tensor_count": tensors,
        "nonfinite": nonfinite,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
