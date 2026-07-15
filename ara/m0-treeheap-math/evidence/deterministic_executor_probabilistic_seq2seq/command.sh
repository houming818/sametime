#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python ara/m0-treeheap-math/src/deterministic_executor_probabilistic_seq2seq_probe.py \
  --out ara/m0-treeheap-math/evidence/deterministic_executor_probabilistic_seq2seq \
  --operator-trials 10000 --sample-trials 10000 --decode-repeats 1000 --seed 67
