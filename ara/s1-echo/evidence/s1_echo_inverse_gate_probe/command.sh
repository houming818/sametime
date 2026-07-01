#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_echo_inverse_gate_probe.py \
  --out ara/s1-echo/evidence/s1_echo_inverse_gate_probe \
  --host-label io.grepcode.cn
