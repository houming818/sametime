#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
OUT=ara/s1-echo/evidence/s1_wmt_streaming_full_queue
mkdir -p "$OUT"
chmod +x ara/s1-echo/src/s1_wmt_streaming_full_queue.sh
nohup bash ara/s1-echo/src/s1_wmt_streaming_full_queue.sh "$OUT" > "$OUT/nohup.out" 2>&1 &
echo $! > "$OUT/pid"
cat "$OUT/pid"
