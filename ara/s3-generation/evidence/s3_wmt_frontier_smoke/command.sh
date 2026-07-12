#!/usr/bin/env bash
set -euo pipefail

python3 ara/s3-generation/src/s3_wmt_frontier_bottleneck.py \
  --data /mnt/nas/datasets/wmt17/train.zh-en \
  --spm-model /mnt/nas/datasets/wmt17/sp_bpe.model \
  --evidence-dir ara/s3-generation/evidence/s3_wmt_frontier_smoke \
  --model all \
  --train-samples 5000 \
  --valid-samples 500 \
  --test-samples 500 \
  --max-scan 100000 \
  --min-len 9 \
  --max-len 24 \
  --dim 192 \
  --hidden 192 \
  --batch-size 32 \
  --epochs 5 \
  --lr 0.002 \
  --device cuda \
  --num-workers 0
