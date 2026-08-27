#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_recursive_depth_pressure_protocol_training.py"
root="ara/s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07"
checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"

test -f "$runner"
test -f "$checkpoint"
mkdir -p "$root"

nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$root/gpu_before.txt"

python3 "$runner" \
  --evidence-dir "$root/self_test" \
  --self-test

python3 "$runner" \
  --checkpoint "$checkpoint" \
  --evidence-dir "$root/smoke_seed10701" \
  --mode smoke \
  --seed 10701 \
  --device cuda \
  2>&1 | tee "$root/smoke_seed10701.log"

python3 - "$root/smoke_seed10701/summary.json" <<'PY'
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
required = {"P0", "P1", "P2", "P3", "P4"}
if set(row["gates"]) != required:
    raise SystemExit("D07 smoke evidence contract is incomplete")
if not row["gates"]["P0"]:
    raise SystemExit("D07 implementation contract failed")
print(json.dumps({"event": "smoke_gate", "decision": row["decision"], "gates": row["gates"]}))
PY

nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$root/gpu_after.txt"

