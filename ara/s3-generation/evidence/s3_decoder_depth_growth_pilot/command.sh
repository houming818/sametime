#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

python3 -u ara/s3-generation/src/s3_decoder_depth_growth.py \
  --data-root /home/nio/datasets/pretrain \
  --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model \
  --evidence-dir ara/s3-generation/evidence/s3_decoder_depth_growth_pilot \
  --context 128 \
  --future 32 \
  --dim 192 \
  --hidden 192 \
  --batch 32 \
  --eval-batch 32 \
  --steps 6000 \
  --eval-batches 8 \
  --log-every 250 \
  --lr 4e-4 \
  --seed 72063 \
  --device cuda
