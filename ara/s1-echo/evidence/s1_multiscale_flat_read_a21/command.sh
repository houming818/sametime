#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_multiscale_flat_read_a21.py \
  --counts ara/s1-echo/evidence/s1_probability_residual_corpus_scale_a14/formal/counts/train_1000000/context_counts.pt \
  --source-checkpoint ara/s1-echo/evidence/s1_conditional_residual_evidence_fold_a20/formal_seed20261001/evidence_fold_dimension_16.pt \
  --out ara/s1-echo/evidence/s1_multiscale_flat_read_a21/formal_seed20261002 \
  --epochs 200 \
  --batch-size 64 \
  --eval-epochs 1,5,10,20,50,100,150,200 \
  --seed 20261002
