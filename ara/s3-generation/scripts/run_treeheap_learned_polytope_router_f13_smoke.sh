#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

evidence="ara/s3-generation/evidence/s3_treeheap_learned_polytope_router_f13/smoke_seed12401"
mkdir -p "$evidence"

command=(
  python3 ara/s3-generation/src/s3_treeheap_learned_polytope_router_f13.py
  --checkpoint ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/treeheap-106m/checkpoint_latest.pt
  --evidence-dir "$evidence"
  --device cuda
  --seed 12401
  --groups 8
  --rank 16
  --steps 300
  --batch-size 8
  --lr 0.002
  --eval-rows 16
  --generation-examples 16
  --max-generation 64
  --log-every 50
)

printf '%q ' "${command[@]}" >"$evidence/command.txt"
printf '\n' >>"$evidence/command.txt"
sha256sum \
  ara/s3-generation/src/s3_treeheap_learned_polytope_router_f13.py \
  ara/s3-generation/logic/treeheap_learned_polytope_router_f13.zh.md \
  >"$evidence/inputs.sha256"

nvidia-smi \
  --query-gpu=timestamp,power.limit,power.draw,temperature.gpu,memory.used,utilization.gpu \
  --format=csv -l 10 >"$evidence/gpu_samples.csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

"${command[@]}" 2>&1 | tee "$evidence/run.log"
