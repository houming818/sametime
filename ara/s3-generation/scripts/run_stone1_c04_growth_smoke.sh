#!/usr/bin/env bash
# Owner: Nio Log Squad
# Author: OpenAI Codex
# Created: 2026-07-23
# Updated: 2026-07-23
# Purpose: Run the short C04 trajectory smoke on io.
set -uo pipefail

ROOT=/home/nio/log/holds/SameTime
EVIDENCE="$ROOT/ara/s3-generation/evidence/s3_stone1_protocol_growth_trajectory_smoke_20260723a"
mkdir -p "$EVIDENCE"
python3 "$ROOT/ara/s3-generation/src/test_stone1_protocol_growth_trajectory.py" >"$EVIDENCE/unit_tests.log" 2>&1 || exit $?
python3 "$ROOT/ara/s3-generation/src/s3_stone1_protocol_growth_trajectory.py" \
  --evidence-dir "$EVIDENCE" --smoke --code-commit "$(git -C "$ROOT" rev-parse HEAD)" \
  2>&1 | tee "$EVIDENCE/stdout.log"
status=${PIPESTATUS[0]}
printf '{"completed":"%s","exit_code":%d}\n' "$(date --iso-8601=seconds)" "$status" >"$EVIDENCE/runner_status.json"
exit "$status"
