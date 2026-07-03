#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_local_flip_echo_probe.py \
  --out ara/s1-echo/evidence/s1_local_flip_echo_probe \
  --samples 20000 \
  --scan-lines 800000 \
  --vocab-size 8192 \
  --min-len 8 \
  --max-len 32 \
  --min-span-len 2 \
  --max-span-len 8 \
  --epochs 18 \
  --batch 256 \
  --eval-batch 512 \
  --dim 128 \
  --lr 0.08 \
  --state-weight 2.0 \
  --entropy-weight 0.005 \
  --device cuda \
  --host-label io.grepcode.cn
