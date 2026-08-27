#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_recursive_depth_length_collapse.py"
root="ara/s3-generation/evidence/s3_recursive_depth_length_collapse_d04"
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
  --checkpoint "$seed10101" \
  --evidence-dir "$root/smoke" \
  --mode smoke \
  --eval-rows 64 \
  --batch-size 8 \
  --device cuda \
  2>&1 | tee "$root/smoke.log"

python3 - "$root/smoke/summary.json" <<'PY'
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not row["gates"]["P0"]:
    raise SystemExit("D04 smoke P0 failed")
if row["seeds"] != [10101]:
    raise SystemExit("D04 smoke seed mismatch")
PY

python3 "$runner" \
  --checkpoint "$seed10101" \
  --checkpoint "$seed10102" \
  --checkpoint "$seed10103" \
  --evidence-dir "$root/formal" \
  --mode formal \
  --eval-rows 256 \
  --batch-size 8 \
  --device cuda \
  2>&1 | tee "$root/formal.log"

python3 - "$root/formal/summary.json" <<'PY'
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if not row["gates"]["P0"]:
    raise SystemExit("D04 formal P0 failed")
print(json.dumps({
    "event": "D04_complete",
    "gates": row["gates"],
    "decision": row["decision"],
}, ensure_ascii=False))
PY
