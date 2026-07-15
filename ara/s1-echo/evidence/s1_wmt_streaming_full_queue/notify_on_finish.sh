#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
E=ara/s1-echo/evidence/s1_wmt_streaming_full_queue
PID=$(cat "$E/pid" 2>/dev/null || true)
if [[ -z "${PID}" ]]; then
  echo "missing pid" > "$E/notify_error.log"
  exit 0
fi
while kill -0 "$PID" 2>/dev/null; do
  sleep 300
done
REPORT="$E/finish_report.txt"
{
  echo "S1 WMT streaming full queue finished at $(date -Is)"
  echo "evidence: /home/nio/log/holds/SameTime/$E"
  echo
  echo "--- queue_summary.json ---"
  cat "$E/queue_summary.json" 2>/dev/null || echo "queue_summary.json not found"
  echo
  echo "--- queue stdout tail ---"
  tail -n 80 "$E/queue_stdout.log" 2>/dev/null || true
  echo
  echo "--- queue stderr tail ---"
  tail -n 80 "$E/queue_stderr.log" 2>/dev/null || true
} > "$REPORT"
if command -v sendme >/dev/null 2>&1; then
  sendme -s "S1 WMT streaming queue finished" -f "$REPORT" || true
fi
