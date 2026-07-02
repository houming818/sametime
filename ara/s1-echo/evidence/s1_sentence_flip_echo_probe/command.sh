#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_sentence_flip_echo_probe.py \
  --out ara/s1-echo/evidence/s1_sentence_flip_echo_probe \
  --samples 20000 \
  --scan-lines 800000 \
  --vocab-size 8192 \
  --min-len 3 \
  --max-len 32 \
  --host-label io.grepcode.cn
