#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_wmt_streaming_full_smoke.py \
  --out "ara/s1-echo/evidence/s1_wmt_streaming_full_queue/bow" \
  --wmt-path "/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv" \
  --models "bow" \
  --steps "60000" \
  --log-every "2000" \
  --batch "256" \
  --max-len "48" \
  --dim "128" \
  --device cuda
