#!/usr/bin/env bash
# Owner: SameTime S3 generation.
# Authors: Houming818 and Codex Review.
# Created: 2026-07-21.
# Updated: 2026-07-21.
# Purpose: Run the formal STONE-1 GPU queue with a single-process lock.
set -euo pipefail

repo="${STONE1_REPO:-/home/nio/log/holds/SameTime}"
evidence="${STONE1_EVIDENCE:-ara/s3-generation/evidence/s3_stone1_private_protocol}"
checkpoint_dir="${STONE1_CHECKPOINT_DIR:-$evidence/checkpoints}"
code_commit="${STONE1_CODE_COMMIT:-4c3d275}"

cd "$repo"
mkdir -p "$evidence" "$checkpoint_dir"
exec 9>/tmp/sametime-stone1-gpu.lock
if ! flock -n 9; then
  echo "STONE-1 refused: another holder owns /tmp/sametime-stone1-gpu.lock" >&2
  exit 73
fi

started="$(date --iso-8601=seconds)"
status=0
python3 ara/s3-generation/src/s3_stone1_private_protocol.py \
  --evidence-dir "$evidence" \
  --checkpoint-dir "$checkpoint_dir" \
  --train-samples 1000000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --fixed-steps 15625 \
  --eval-interval 500 \
  --seeds 71901 71902 71903 \
  --variants identity learned_structural frozen_random \
  --code-commit "$code_commit" \
  >"$evidence/stdout.log" 2>"$evidence/stderr.log" || status=$?

completed="$(date --iso-8601=seconds)"
printf '{"started":"%s","completed":"%s","exit_code":%d}\n' \
  "$started" "$completed" "$status" >"$evidence/runner_status.json"

if command -v sendme >/dev/null 2>&1; then
  if [[ "$status" -eq 0 ]]; then
    sendme -s "STONE-1 completed" "STONE-1 formal run completed on $(hostname)." || true
  else
    sendme -s "STONE-1 failed" "STONE-1 formal run failed on $(hostname), exit=$status." || true
  fi
fi
exit "$status"
