#!/usr/bin/env bash
set -uo pipefail

repo=/home/nio/log/holds/SameTime
evidence="$repo/ara/s3-generation/evidence/s3_decoder_depth_growth_pilot"
mkdir -p "$evidence"
cd "$repo"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

log "launcher started; waiting for the current full-corpus repair job"
idle_checks=0
while (( idle_checks < 2 )); do
  if pgrep -f '[s]3_full_corpus_repair_seq2seq.py' >/dev/null; then
    idle_checks=0
    log "current repair training is still running"
  elif nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    idle_checks=0
    log "GPU still has a compute process"
  else
    idle_checks=$((idle_checks + 1))
    log "GPU idle confirmation ${idle_checks}/2"
  fi
  sleep 60
done

log "GPU is idle; recording safety state and starting depth-growth pilot"
nvidia-smi --query-gpu=timestamp,name,power.draw,power.limit,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$evidence/gpu_before.csv" 2>&1 || true

bash "$evidence/command.sh" > "$evidence/stdout.log" 2> "$evidence/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$evidence/exit_code.txt"
nvidia-smi --query-gpu=timestamp,name,power.draw,power.limit,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$evidence/gpu_after.csv" 2>&1 || true

if (( rc == 0 )); then
  log "depth-growth pilot completed successfully"
  command -v sendme >/dev/null && sendme -s "TreeHeap depth-growth completed" \
    "The overnight pilot completed. Evidence: $evidence" || true
else
  log "depth-growth pilot failed with exit code $rc"
  command -v sendme >/dev/null && sendme -s "TreeHeap depth-growth failed" \
    "The overnight pilot failed with rc=$rc. Evidence: $evidence" || true
fi

exit "$rc"
