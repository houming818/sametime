#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

root="ara/s3-generation/evidence/s3_predictive_state_sketch_d12/smoke_seed11201"
runner="ara/s3-generation/src/s3_predictive_state_sketch_d12.py"
cpu_test="ara/s3-generation/src/test_predictive_state_sketch_d12.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001/task/checkpoint_best.pt"

test -f "$runner"
test -f "$cpu_test"
test -f "$source_checkpoint"
test -f "$warm_start"

python3 "$cpu_test"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

mkdir -p "$root"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_before.csv"

python3 "$runner" \
  --mode smoke --steps 500 --batch-size 16 --eval-rows 128 \
  --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
  --evidence-dir "$root" --seed 11201 --ownership-seed 11202 \
  --sketch-seed 11203 --sketch-width 128 --predictive-weight 0.10 \
  --device cuda 2>&1 | tee "$root/run.log"

nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_after.csv"

sendme -s "D12 predictive-state sketch smoke finished" -f "$root/comparison.json" || true
