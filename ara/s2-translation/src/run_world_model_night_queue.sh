#!/usr/bin/env bash
set -euo pipefail

# Single-task night queue for the fragile RTX 3090 on io.grepcode.cn.
# It never changes power/clock limits. It records snapshots and runs one
# training process only after the GPU is clear.

REPO_ROOT="${REPO_ROOT:-/data/homecicd/sametime}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/homecicd/sametime/ara/s2-translation/evidence/world_model_night_$(date +%Y%m%d_%H%M%S)}"
NAS_ROOT="${NAS_ROOT:-/mnt/nas/datasets/wmt_massive/evidence_nio/world_model_night_$(date +%Y%m%d_%H%M%S)}"
LOCK_FILE="${LOCK_FILE:-/tmp/nio_world_model_night.lock}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"
TRAIN_HOURS="${TRAIN_HOURS:-10}"
TRAIN_MAX_LINES="${TRAIN_MAX_LINES:-500000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-3072}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-1200}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"

mkdir -p "$LOCAL_ROOT"
LOG="$LOCAL_ROOT/queue.log"
STATUS="$LOCAL_ROOT/status.txt"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,power.limit,power.draw,clocks.gr,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv \
    | tee -a "$LOCAL_ROOT/gpu_queue_snapshots.csv" >/dev/null || true
}

active_experiment_pids() {
  pgrep -u "$(id -u)" -f 's2_overnight_io.py|frame_probe.py|train_world_model_night.py|anchor_tree|torchrun' || true
}

wait_for_slot() {
  local waited=0
  while true; do
    local pids
    pids="$(active_experiment_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [[ -z "$pids" ]]; then
      log "slot free"
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

main() {
  {
    flock -n 9 || {
      echo "another world-model queue is already active" | tee -a "$LOG"
      exit 3
    }
    echo "running" > "$STATUS"
    log "queue acquired"
    log "will not change GPU power or clock limits"
    log "LOCAL_ROOT=$LOCAL_ROOT"
    log "NAS_ROOT=$NAS_ROOT"
    gpu_snapshot
    wait_for_slot
    cd "$REPO_ROOT"
    log "starting train_world_model_night.py"
    CUDA_VISIBLE_DEVICES=0 python3 ara/s2-translation/src/train_world_model_night.py \
      --local-root "$LOCAL_ROOT" \
      --nas-root "$NAS_ROOT" \
      --hours "$TRAIN_HOURS" \
      --max-lines "$TRAIN_MAX_LINES" \
      --batch-size "$TRAIN_BATCH_SIZE" \
      --steps-per-epoch "$TRAIN_STEPS_PER_EPOCH" \
      --epochs "$TRAIN_EPOCHS" \
      --device cuda
    echo "done" > "$STATUS"
    log "queue done"
  } 9>"$LOCK_FILE"
}

main "$@"
