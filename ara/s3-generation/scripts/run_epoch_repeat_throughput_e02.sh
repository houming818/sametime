#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

root="ara/s3-generation/evidence/s3_epoch_repeat_throughput_e02/formal_seed11302"
runner="ara/s3-generation/src/s3_epoch_repeat_scaling_prefetch.py"
compare="ara/s3-generation/src/s3_epoch_repeat_throughput_e02_compare.py"
test_runner="ara/s3-generation/src/test_epoch_repeat_scaling_prefetch.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001/task/checkpoint_best.pt"
resume="ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11/formal_seed11101/treeheap-106m/checkpoint_latest.pt"

python3 "$test_runner"
mkdir -p "$root"

run_case() {
  local name="$1"
  local batch="$2"
  local prefetch="$3"
  local output="$root/$name"
  rm -rf "$output"
  mkdir -p "$output"

  local power_limit
  power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
  python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

  nvidia-smi \
    --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader --loop=2 > "$output/gpu_samples.csv" &
  local sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' RETURN

  python3 "$runner" \
    --prefetch-batches "$prefetch" \
    --mode formal --arm treeheap-106m --extra-dim 384 \
    --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
    --resume-checkpoint "$resume" --evidence-dir "$output" \
    --batch-size "$batch" --max-segment-steps 200 --target-cursor 7304358 \
    --expected-start-cursor 400488 --corpus-pass 2 --direction-flip 0 \
    --seed 11101 --ownership-seed 11102 --eval-rows 32 \
    --generation-examples 8 --log-every 25 --device cuda \
    2>&1 | tee "$output/run.log"

  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  trap - RETURN
  python3 - "$output/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert all(x["gates"].values()), x["gates"]
assert x["step"] - x["start_step"] == 200
assert x["cursor"] > x["start_cursor"]
PY
  rm -f "$output/checkpoint_latest.pt" "$output/checkpoint_best.pt"
}

run_case b64-raw 64 0
run_case b64-prefetch2 64 2
run_case b96-prefetch2 96 2
run_case b128-prefetch2 128 2

python3 "$compare" "$root" | tee "$root/compare.log"
sendme -s "Epoch repeat bounded throughput E02 completed" -f "$root/comparison.json" || true
