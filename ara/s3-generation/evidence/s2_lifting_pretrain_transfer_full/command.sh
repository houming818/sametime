#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s2_lifting_pretrain_transfer.py \
  --evidence-dir ara/s3-generation/evidence/s2_lifting_pretrain_transfer_full \
  --pretrain-steps 50000 \
  --pretrain-valid-every 2500 \
  --pretrain-valid-batches 16 \
  --wmt-train-samples 200000 \
  --wmt-valid-samples 5000 \
  --wmt-test-samples 5000 \
  --wmt-max-scan 2000000 \
  --wmt-epochs 5 \
  --num-workers 2
