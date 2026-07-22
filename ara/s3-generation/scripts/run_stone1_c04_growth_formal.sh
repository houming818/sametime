#!/usr/bin/env bash
# Owner: Nio Log Squad
# Author: OpenAI Codex
# Created: 2026-07-23
# Updated: 2026-07-23
# Purpose: Run the preregistered single-seed C04 growth trajectory on io.
set -uo pipefail

ROOT=/home/nio/log/holds/SameTime
EVIDENCE="$ROOT/ara/s3-generation/evidence/s3_stone1_protocol_growth_trajectory"
CODE_COMMIT=26c8269
mkdir -p "$EVIDENCE"
if [[ -e "$EVIDENCE/summary.json" ]]; then
  echo "refusing to overwrite completed evidence: $EVIDENCE" >&2
  exit 2
fi
python3 "$ROOT/ara/s3-generation/src/test_stone1_protocol_growth_trajectory.py" >"$EVIDENCE/unit_tests.log" 2>&1 || exit $?
python3 "$ROOT/ara/s3-generation/src/s3_stone1_protocol_growth_trajectory.py" \
  --evidence-dir "$EVIDENCE" --code-commit "$CODE_COMMIT" \
  2>&1 | tee "$EVIDENCE/stdout.log"
status=${PIPESTATUS[0]}
printf '{"completed":"%s","exit_code":%d}\n' "$(date --iso-8601=seconds)" "$status" >"$EVIDENCE/runner_status.json"
exit "$status"
