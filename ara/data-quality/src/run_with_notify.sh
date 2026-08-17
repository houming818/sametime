#!/usr/bin/env bash
set -uo pipefail

subject="$1"
shift
log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT

"$@" 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}

if [[ $status -eq 0 ]]; then
  tail -n 80 "$log_file" | sendme -s "$subject completed" || true
else
  {
    echo "Command failed with exit code $status"
    echo
    tail -n 120 "$log_file"
  } | sendme -s "$subject FAILED" || true
fi

exit "$status"
