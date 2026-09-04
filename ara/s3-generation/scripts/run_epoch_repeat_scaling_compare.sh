#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
root="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301"
python3 ara/s3-generation/src/s3_epoch_repeat_scaling_compare.py --root "$root" \
  | tee "$root/compare.log"
sendme -s "Epoch Repeat Scaling paired comparison completed" \
  -f "$root/comparison.json" || true

