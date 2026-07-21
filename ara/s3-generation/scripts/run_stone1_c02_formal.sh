#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/nio/log/holds/SameTime
EVIDENCE="$ROOT/ara/s3-generation/evidence/s3_stone1_canonical_codec"
CODE_COMMIT=decc78e

mkdir -p "$EVIDENCE/checkpoints"
if [[ -e "$EVIDENCE/summary.json" ]]; then
  echo "refusing to overwrite completed formal evidence: $EVIDENCE" >&2
  exit 2
fi

started=$(date --iso-8601=seconds)
python3 "$ROOT/ara/s3-generation/src/test_stone1_canonical_codec.py" \
  >"$EVIDENCE/unit_tests.log" 2>&1
unit_status=$?
if [[ $unit_status -ne 0 ]]; then
  printf '{"started":"%s","completed":"%s","exit_code":%d,"stage":"unit_tests"}\n' \
    "$started" "$(date --iso-8601=seconds)" "$unit_status" >"$EVIDENCE/runner_status.json"
  exit "$unit_status"
fi

python3 "$ROOT/ara/s3-generation/src/s3_stone1_canonical_codec.py" \
  --evidence-dir "$EVIDENCE" \
  --checkpoint-dir "$EVIDENCE/checkpoints" \
  --train-samples 1000000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --fixed-steps 15625 \
  --eval-interval 500 \
  --seeds 71901 71902 71903 \
  --variants canonical_algebraic canonical_learned canonical_frozen \
  --code-commit "$CODE_COMMIT" \
  >"$EVIDENCE/stdout.log" 2>"$EVIDENCE/stderr.log"
status=$?
printf '{"started":"%s","completed":"%s","exit_code":%d,"stage":"formal"}\n' \
  "$started" "$(date --iso-8601=seconds)" "$status" >"$EVIDENCE/runner_status.json"
exit "$status"
