#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_recursive_depth_pressure_protocol_training.py"
root="ara/s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07r2"
checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"

test -f "$runner"
test -f "$checkpoint"
mkdir -p "$root"
nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$root/gpu_before.txt"

python3 "$runner" \
  --checkpoint "$checkpoint" \
  --evidence-dir "$root/smoke_seed10721" \
  --mode smoke \
  --seed 10721 \
  --steps 600 \
  --log-every 120 \
  --freeze-language-backbone \
  --bounded-protocol-gain \
  --device cuda \
  2>&1 | tee "$root/smoke_seed10721.log"

python3 - "$root/smoke_seed10721/summary.json" <<'PY'
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if row["claim"] != "S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R2":
    raise SystemExit("wrong D07R2 claim")
if not row["contracts"]["bounded_protocol_gain"] or not row["gates"]["P0"]:
    raise SystemExit("D07R2 gain contract failed")
print(json.dumps({
    "event": "r2_smoke_gate", "decision": row["decision"],
    "gates": row["gates"], "gain": row["protocol_gain_final"],
}))
PY

nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$root/gpu_after.txt"

