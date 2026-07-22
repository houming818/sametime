#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/nio/log/holds/SameTime
RUNNER="$ROOT/ara/s3-generation/scripts/run_stone1_c03_capacity_formal.sh"
EVIDENCE="$ROOT/ara/s3-generation/evidence/s3_stone1_capacity_rate_distortion"
LAUNCH_LOG=/tmp/s3_stone1_c03_capacity_launcher.log

if [[ -e "$EVIDENCE/summary.json" ]]; then
  echo "formal evidence is already complete: $EVIDENCE/summary.json"
  exit 0
fi
if pgrep -af "[r]un_stone1_c03_capacity_formal.sh" >/dev/null; then
  echo "formal runner is already active"
  pgrep -af "[r]un_stone1_c03_capacity_formal.sh"
  exit 0
fi

mkdir -p "$EVIDENCE"
nohup setsid bash "$RUNNER" >"$LAUNCH_LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$EVIDENCE/launcher.pid"
printf 'pid=%s log=%s evidence=%s\n' "$pid" "$LAUNCH_LOG" "$EVIDENCE"
