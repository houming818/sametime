#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
OUT=ara/s1-echo/evidence/s1_wmt_massive_canonical_full_try
mkdir -p "$OUT"
sed -i 's/\r$//' "$OUT/command.sh" 2>/dev/null || true
nohup bash "$OUT/command.sh" >> "$OUT/stdout.log" 2>> "$OUT/stderr.log" &
echo $! > "$OUT/pid"
cat "$OUT/pid"
