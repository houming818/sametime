#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_lifting_pump_wmt.py \
  --data /mnt/nas/datasets/wmt17/train.zh-en \
  --spm-model /mnt/nas/datasets/wmt17/sp_bpe.model \
  --evidence-dir ara/s3-generation/evidence/s2_lifting_pump_wmt_full \
  --train-samples 27000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --max-scan 300000 \
  --epochs 10 \
  --dim 256 \
  --hidden 256 \
  --batch-size 64 \
  --num-workers 2 \
  --variants target_only flat_seq lifting_root lifting_full lifting_recursive
