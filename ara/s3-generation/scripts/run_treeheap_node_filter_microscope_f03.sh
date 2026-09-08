#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

mode="${1:?mode is required: smoke or formal}"
case "$mode" in
  smoke) output="ara/s3-generation/evidence/s3_treeheap_node_filter_microscope_f03/smoke" ;;
  formal) output="ara/s3-generation/evidence/s3_treeheap_node_filter_microscope_f03/formal" ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

checkpoint="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt"
specimen="ara/s3-generation/data/treeheap_filter_specimens_f03.jsonl"
runner="ara/s3-generation/src/s3_treeheap_node_filter_microscope_f03.py"

test -f "$checkpoint"
test -f "$specimen"
test -f "$runner"
mkdir -p "$output"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

sha256sum "$checkpoint" "$specimen" "$runner" >"$output/inputs.sha256"
printf 'python3 %q --mode %q --checkpoint %q --specimen %q --evidence-dir %q --depth 7 --max-new-tokens 64 --device cuda\n' \
  "$runner" "$mode" "$checkpoint" "$specimen" "$output" >"$output/command.txt"

nvidia-smi \
  --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader --loop=5 >>"$output/gpu_samples.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT

python3 "$runner" \
  --mode "$mode" --checkpoint "$checkpoint" --specimen "$specimen" \
  --evidence-dir "$output" --depth 7 --max-new-tokens 64 --device cuda \
  2>&1 | tee "$output/run.log"

kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
sendme -s "F03 node filter microscope ${mode} completed" -f "$output/summary.json" || true
