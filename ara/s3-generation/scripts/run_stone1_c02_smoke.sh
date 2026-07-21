#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/nio/log/holds/SameTime
EVIDENCE="$ROOT/ara/s3-generation/evidence/s3_stone1_canonical_codec_smoke_20260722a"
CODE_COMMIT=22b5ef7

mkdir -p "$EVIDENCE/checkpoints"
if [[ -e "$EVIDENCE/summary.json" ]]; then
  echo "refusing to overwrite completed smoke evidence: $EVIDENCE" >&2
  exit 2
fi

started=$(date --iso-8601=seconds)
python3 "$ROOT/ara/s3-generation/src/test_stone1_canonical_codec.py" \
  >"$EVIDENCE/unit_tests.log" 2>&1
unit_status=$?
if [[ $unit_status -ne 0 ]]; then
  printf '{"started":"%s","completed":"%s","exit_code":%d,"stage":"unit_tests"}\n' \
    "$started" "$(date --iso-8601=seconds)" "$unit_status" >"$EVIDENCE/runner_status.json"
  cat "$EVIDENCE/unit_tests.log"
  exit "$unit_status"
fi

python3 "$ROOT/ara/s3-generation/src/s3_stone1_canonical_codec.py" \
  --evidence-dir "$EVIDENCE" \
  --checkpoint-dir "$EVIDENCE/checkpoints" \
  --smoke \
  --code-commit "$CODE_COMMIT" \
  >"$EVIDENCE/stdout.log" 2>"$EVIDENCE/stderr.log"
status=$?
printf '{"started":"%s","completed":"%s","exit_code":%d,"stage":"smoke"}\n' \
  "$started" "$(date --iso-8601=seconds)" "$status" >"$EVIDENCE/runner_status.json"
cat "$EVIDENCE/unit_tests.log"
tail -n 80 "$EVIDENCE/stdout.log"
if [[ $status -ne 0 ]]; then
  cat "$EVIDENCE/stderr.log" >&2
fi
exit "$status"
