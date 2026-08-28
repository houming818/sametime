#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_structural_slot_ownership_d08.py"
root="ara/s3-generation/evidence/s3_structural_slot_ownership_d08"
run="$root/smoke_r1_seed10801"
checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"

test -f "$runner"
test -f "$checkpoint"
mkdir -p "$root"

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_before.csv"

python3 "$runner" \
  --evidence-dir "$root/selftest" \
  --self-test

rm -rf "$run"
python3 "$runner" \
  --checkpoint "$checkpoint" \
  --evidence-dir "$run" \
  --seed 10801 \
  --steps 600 \
  --batch-size 8 \
  --train-rows 2048 \
  --eval-rows 128 \
  --max-slots 32 \
  --log-every 120 \
  --device cuda \
  2>&1 | tee "$root/smoke_r1_seed10801.log"

python3 - "$run/summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["claim"] == "S3-STRUCTURAL-SLOT-OWNERSHIP-D08R1"
assert payload["rows"] == {"train": 2048, "valid": 128, "test": 128}
assert payload["initialization_match"] is True
assert set(payload["arms"]) == {"free", "subheap", "random"}
for arm in payload["arms"].values():
    assert arm["source_sha256_before"] == arm["source_sha256_after"]
    assert arm["language_sha256_before"] == arm["language_sha256_after"]
    for depth in (5, 6, 7):
        row = arm["final_test"][str(depth)]["native"]
        assert math.isfinite(row["nll"])
        assert math.isfinite(row["slot_variance"])
print(json.dumps({
    "event": "evidence_validated",
    "decision": payload["decision"],
    "gates": payload["gates"],
}, ensure_ascii=False))
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_after.csv"
