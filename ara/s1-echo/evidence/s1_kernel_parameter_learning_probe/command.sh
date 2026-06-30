#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_kernel_parameter_learning_probe.py \
  --out ara/s1-echo/evidence/s1_kernel_parameter_learning_probe \
  --host-label io.grepcode.cn
