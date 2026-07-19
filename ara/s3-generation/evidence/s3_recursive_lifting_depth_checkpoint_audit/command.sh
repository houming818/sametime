#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s3-generation/src/s3_recursive_lifting_depth_checkpoint_audit.py \
  --checkpoint ara/s3-generation/evidence/s2_adaptive_lifting_wmt_200k/checkpoint_learned_update.pt \
  --evidence-dir ara/s3-generation/evidence/s3_recursive_lifting_depth_checkpoint_audit \
  --examples 8
