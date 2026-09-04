#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

arm="${1:?arm is required}"
case "$arm" in
  treeheap-63m) extra_dim=0 ;;
  treeheap-106m) extra_dim=384 ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

root="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301"
output="$root/$arm"
runner="ara/s3-generation/src/s3_epoch_repeat_scaling.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001/task/checkpoint_best.pt"
d11_checkpoint="ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11/formal_seed11101/$arm/checkpoint_latest.pt"

test -f "$runner"
test -f "$source_checkpoint"
test -f "$warm_start"
test -f "$d11_checkpoint"
mkdir -p "$output"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

nvidia-smi \
  --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader --loop=10 > "$output/gpu_samples.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT

resume="$d11_checkpoint"
expected_cursor=400488
segment=0
while true; do
  segment=$((segment + 1))
  if [[ "$segment" -gt 20 ]]; then
    echo "segment guard exceeded" >&2
    exit 4
  fi
  power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
  python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY
  python3 "$runner" \
    --mode formal --arm "$arm" --extra-dim "$extra_dim" \
    --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
    --resume-checkpoint "$resume" --evidence-dir "$output" \
    --batch-size 64 --max-segment-steps 10000 --target-cursor 7304358 \
    --expected-start-cursor "$expected_cursor" --corpus-pass 2 \
    --direction-flip 0 --seed 11101 --ownership-seed 11102 \
    --eval-rows 512 --generation-examples 64 --log-every 500 --device cuda \
    2>&1 | tee -a "$output/run.log"
  python3 - "$output/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
for name, passed in x["gates"].items():
    assert passed, f"safety gate failed: {name}"
assert x["cursor"] > x["start_cursor"]
PY
  complete="$(python3 - "$output/summary.json" <<'PY'
import json, sys
print("yes" if json.load(open(sys.argv[1], encoding="utf-8"))["completed"] else "no")
PY
)"
  if [[ "$complete" == "yes" ]]; then
    break
  fi
  resume="$output/checkpoint_latest.pt"
  expected_cursor="$(python3 - "$output/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["cursor"])
PY
)"
done

kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
sendme -s "Epoch Repeat Scaling $arm completed" -f "$output/summary.json" || true

