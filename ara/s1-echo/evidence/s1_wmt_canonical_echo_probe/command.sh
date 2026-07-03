#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_wmt_canonical_echo_probe.py \
  --out ara/s1-echo/evidence/s1_wmt_canonical_echo_probe \
  --samples 50000 \
  --scan-lines 500000 \
  --min-len 4 \
  --max-len 48 \
  --en-vocab 8192 \
  --zh-vocab 8192 \
  --epochs 5 \
  --batch 256 \
  --eval-batch 256 \
  --max-eval 2000 \
  --dim 128 \
  --lr 0.002 \
  --align-weight 1.0 \
  --echo-weight 1.0 \
  --var-weight 1.0 \
  --target-std 0.05 \
  --min-random-gain 5.0 \
  --min-echo-token-acc 0.55 \
  --device cuda \
  --host-label io.grepcode.cn
