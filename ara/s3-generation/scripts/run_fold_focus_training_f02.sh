#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

mode="${1:?mode is required: smoke or formal}"
case "$mode" in
  smoke)
    output="ara/s3-generation/evidence/s3_fold_focus_training_f02/smoke_r1_seed11401"
    timeout_label="smoke"
    ;;
  formal)
    output="ara/s3-generation/evidence/s3_fold_focus_training_f02/formal_seed11401"
    timeout_label="formal"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

checkpoint="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt"
specimens="ara/s3-generation/data/fold_microscope_specimens_f01.jsonl"
runner="ara/s3-generation/src/s3_fold_focus_training_f02.py"

test -f "$checkpoint"
test -f "$specimens"
test -f "$runner"
mkdir -p "$output"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

nvidia-smi \
  --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader --loop=10 >> "$output/gpu_samples.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT

resume=()
if [[ -f "$output/focus_checkpoint_latest.pt" ]]; then
  resume=(--resume)
fi

python3 "$runner" \
  --mode "$mode" --checkpoint "$checkpoint" --specimens "$specimens" \
  --evidence-dir "$output" --seed 11401 --batch-size 64 \
  --max-steps 5000 --wake-every 500 --log-every 50 --lr 0.001 \
  --eval-rows 256 --generation-examples 32 --max-generation 64 \
  --max-lines 500000 --device cuda "${resume[@]}" \
  2>&1 | tee -a "$output/run.log"

python3 - "$output/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
assert all(x["gates"].values()), x["gates"]
PY

kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
sendme -s "F02 focus training ${timeout_label} completed" -f "$output/summary.json" || true
