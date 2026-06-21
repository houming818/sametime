#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p evidence/existence_proof_suite

python3 src/existence_proof_suite.py \
  --profile night \
  --time-budget-hours "${TIME_BUDGET_HOURS:-8}" \
  --seed "${SEED:-20260621}" \
  --device "${DEVICE:-cuda}" \
  --out-dir evidence/existence_proof_suite
