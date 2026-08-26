#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_recursive_depth_probability_exposure.py"
root="ara/s3-generation/evidence/s3_recursive_depth_probability_exposure_d03"
seed10101="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
seed10102="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12r1/formal_seed10102/read/checkpoint_best.pt"
seed10103="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12r1/formal_seed10103/read/checkpoint_best.pt"

test -f "$runner"
test -f "$seed10101"
test -f "$seed10102"
test -f "$seed10103"
mkdir -p "$root"
nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader

python3 "$runner" \
  --self-test \
  --evidence-dir "$root/self_test" \
  --device cpu

python3 "$runner" \
  --checkpoint "$seed10101" \
  --evidence-dir "$root/smoke" \
  --mode smoke \
  --eval-rows 32 \
  --batch-size 4 \
  --conditions native pair_break_depth_0 \
  --device cuda \
  2>&1 | tee "$root/smoke.log"

python3 - "$root/smoke/summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
row = json.loads(path.read_text(encoding="utf-8"))
if not row["gates"]["P0"]:
    raise SystemExit("D03 smoke P0 failed")
if row["seeds"] != [10101]:
    raise SystemExit("D03 smoke seed mismatch")
PY

python3 "$runner" \
  --checkpoint "$seed10101" \
  --checkpoint "$seed10102" \
  --checkpoint "$seed10103" \
  --evidence-dir "$root/formal" \
  --mode formal \
  --eval-rows 256 \
  --batch-size 8 \
  --conditions native runtime_identity pair_break_depth_0 source_shuffle \
  --device cuda \
  2>&1 | tee "$root/formal.log"

python3 - "$root/formal/summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
row = json.loads(path.read_text(encoding="utf-8"))
if not row["gates"]["P0"]:
    raise SystemExit("D03 formal P0 failed")
if row["seeds"] != [10101, 10102, 10103]:
    raise SystemExit("D03 formal seed mismatch")
print(json.dumps({
    "event": "D03_complete",
    "gates": row["gates"],
    "decision": row["decision"],
}, ensure_ascii=False))
PY
