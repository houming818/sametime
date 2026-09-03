#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

claim_root="ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11"
runner="ara/s3-generation/src/s3_structural_protocol_capacity_ladder_d11.py"
compare="ara/s3-generation/src/s3_structural_protocol_capacity_ladder_d11_compare.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
d10_root="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001"
warm_start="$d10_root/task/checkpoint_best.pt"

test -f "$runner"
test -f "$compare"
test -f "$source_checkpoint"
test -f "$warm_start"
test -f "$d10_root/recovery_summary.json"

python3 - "$d10_root/recovery_summary.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["complete"], "D10 recovery is not complete"
for gate in ("data_cursor_complete", "input_causality", "structure", "source_frozen", "reload"):
    assert x["gates"][gate], f"D10 handoff gate failed: {gate}"
PY

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

common=(
  --source-checkpoint "$source_checkpoint"
  --warm-start "$warm_start"
  --seed 11101 --ownership-seed 11102
  --batch-size 16 --device cuda
)

smoke="$claim_root/smoke_seed11101"
mkdir -p "$smoke"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$smoke/gpu_before.csv"
python3 "$runner" --mode smoke --arm treeheap-63m --extra-dim 0 \
  --evidence-dir "$smoke/treeheap-63m" "${common[@]}" \
  2>&1 | tee "$smoke/treeheap-63m.log"
python3 "$runner" --mode smoke --arm treeheap-106m --extra-dim 384 \
  --evidence-dir "$smoke/treeheap-106m" "${common[@]}" \
  2>&1 | tee "$smoke/treeheap-106m.log"
python3 "$compare" --root "$smoke" --mode smoke | tee "$smoke/compare.log"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$smoke/gpu_after.csv"

formal="$claim_root/formal_seed11101"
mkdir -p "$formal"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$formal/gpu_before.csv"
python3 "$runner" --mode formal --arm treeheap-63m --extra-dim 0 \
  --max-steps 25000 --wake-every 5000 \
  --evidence-dir "$formal/treeheap-63m" "${common[@]}" \
  2>&1 | tee "$formal/treeheap-63m.log"
python3 "$runner" --mode formal --arm treeheap-106m --extra-dim 384 \
  --max-steps 25000 --wake-every 5000 \
  --evidence-dir "$formal/treeheap-106m" "${common[@]}" \
  2>&1 | tee "$formal/treeheap-106m.log"
python3 "$compare" --root "$formal" --mode formal | tee "$formal/compare.log"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$formal/gpu_after.csv"

sendme -s "D11 TreeHeap 63M vs 106M capacity rung finished" -f "$formal/comparison.json" || true
