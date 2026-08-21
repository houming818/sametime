#!/usr/bin/env python3
"""Gate the full score pass on the registered smoke artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SMOKE = Path("ara/data-quality/evidence/full_parallel_cleaning/smoke_2000_v3")
FORMAL = Path("ara/data-quality/evidence/full_parallel_cleaning/formal_14m_v1")


def main() -> None:
    records = []
    for path in sorted(SMOKE.glob("shard_*.done.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if [item["shard"] for item in records] != [0, 1]:
        raise RuntimeError("smoke did not produce exactly two contiguous shards")
    if sum(item["rows"] for item in records) != 2000:
        raise RuntimeError("smoke row count is not 2000")
    for item in records:
        if not item.get("manifest_sha256") or not item.get("accepted_sha256"):
            raise RuntimeError("smoke shard is missing content identities")
    command = [
        sys.executable,
        "ara/data-quality/src/score_parallel_corpus_full.py",
        "--output", str(FORMAL),
        "--shard-rows", "250000",
    ]
    print(json.dumps({"smoke_gate": "passed", "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
