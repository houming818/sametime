#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

root="ara/s3-generation/evidence/s3_predictive_state_sketch_d12/engine_canary_20260904"
runner="ara/s3-generation/src/s3_predictive_state_sketch_d12.py"
analyzer="ara/s3-generation/src/s3_predictive_state_engine_canary_d12.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001/task/checkpoint_best.pt"

test -f "$runner"
test -f "$analyzer"
test -f "$source_checkpoint"
test -f "$warm_start"
mkdir -p "$root"

for batch in 16 32 64; do
  run="$root/batch_$batch"
  mkdir -p "$run"
  power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
  python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

  nvidia-smi \
    --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader --loop=2 > "$run/gpu_samples.csv" &
  sampler=$!
  set +e
  python3 "$runner" \
    --mode pilot --steps 300 --batch-size "$batch" --eval-rows 32 \
    --generation-examples 4 --max-lines 50000 --log-every 50 \
    --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
    --evidence-dir "$run" --seed 11201 --ownership-seed 11202 \
    --sketch-seed 11203 --sketch-width 128 --predictive-weight 0.53 \
    --device cuda > "$run/run.log" 2>&1
  status=$?
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  printf '%s\n' "$status" > "$run/exit_code.txt"
  if [[ "$status" -ne 0 ]]; then
    break
  fi
done

python3 "$analyzer" --root "$root" | tee "$root/analyze.log"
sendme -s "D12-E1 GPU throughput canary finished" -f "$root/throughput_summary.json" || true
