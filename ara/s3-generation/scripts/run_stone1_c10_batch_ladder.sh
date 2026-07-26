#!/usr/bin/env bash
set -u

cd /home/nio/log/holds/SameTime
root="ara/s3-generation/evidence/s3_stone1_c10_batch_ladder"
mkdir -p "$root"
: > "$root/runner.tsv"

for batch in 8 16 32 64 96 128; do
  evidence="$root/batch_${batch}"
  mkdir -p "$evidence"
  started=$(date +%s)
  if python3 ara/s3-generation/src/s3_stone1_c10_long_smoke.py \
      --evidence-dir "$evidence" \
      --source-width 128 \
      --target-width 128 \
      --heap-width 256 \
      --batch "$batch" \
      --steps 20 \
      --eval-every 20 \
      --valid-batches 1 \
      --amp-dtype bfloat16 \
      2>&1 | tee "$evidence/stdout.log"; then
    status=done
  else
    status=failed
  fi
  elapsed=$(($(date +%s) - started))
  printf '%s\t%s\t%s\n' "$batch" "$status" "$elapsed" | tee -a "$root/runner.tsv"
  if [[ "$status" == failed ]]; then
    break
  fi
done
