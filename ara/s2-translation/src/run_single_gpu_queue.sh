#!/usr/bin/env bash
set -euo pipefail

# Single-GPU queue runner for io.grepcode.cn.
#
# The RTX 3090 on io is treated as a fragile, power-limited card. This script
# does not change clocks or power limits. It only records the active limit,
# waits for existing GPU experiment processes to finish, and then runs exactly
# one queued job under a lock.

QUEUE_ROOT="${QUEUE_ROOT:-/data/homecicd/sametime/ara/s2-translation/evidence/frame_probe_2h_queue}"
REPO_ROOT="${REPO_ROOT:-/data/homecicd/sametime}"
LOCK_FILE="${LOCK_FILE:-/tmp/nio_single_gpu_queue.lock}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
JOB_NAME="${JOB_NAME:-frame_probe_2h}"

mkdir -p "$QUEUE_ROOT"
LOG="$QUEUE_ROOT/${JOB_NAME}.log"
STATUS="$QUEUE_ROOT/${JOB_NAME}.status"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,power.limit,power.draw,clocks.gr,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv \
    | tee -a "$QUEUE_ROOT/gpu_snapshots.csv" >/dev/null || true
}

existing_experiment_pids() {
  pgrep -u "$(id -u)" -f 's2_overnight_io.py|frame_probe.py|train.*tree|anchor_tree' || true
}

wait_for_slot() {
  local waited=0
  while true; do
    local pids
    pids="$(existing_experiment_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [[ -z "$pids" ]]; then
      log "queue slot free"
      return 0
    fi
    if (( waited >= MAX_WAIT_SECONDS )); then
      log "max wait reached with active pids: $pids"
      return 2
    fi
    log "waiting for active experiment pids: $pids"
    gpu_snapshot
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
}

run_frame_probe() {
  local out="$QUEUE_ROOT/output"
  rm -rf "$out"
  mkdir -p "$out"
  log "starting frame_probe.py"
  gpu_snapshot
  cd "$REPO_ROOT"
  CUDA_VISIBLE_DEVICES=0 python3 ara/s2-translation/src/frame_probe.py \
    --out "$out" \
    --device cuda
  gpu_snapshot
  log "finished frame_probe.py"
}

main() {
  {
    flock -n 9 || {
      echo "another queue worker is already active" | tee -a "$LOG"
      exit 3
    }
    echo "running" > "$STATUS"
    log "single gpu queue acquired"
    log "this runner will not change GPU power or clock limits"
    gpu_snapshot
    wait_for_slot
    run_frame_probe
    echo "done" > "$STATUS"
    log "queue done"
  } 9>"$LOCK_FILE"
}

main "$@"
