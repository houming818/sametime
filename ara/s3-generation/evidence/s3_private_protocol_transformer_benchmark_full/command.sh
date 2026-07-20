#!/usr/bin/env bash
set -uo pipefail

cd /home/nio/log/holds/SameTime
OUT=ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full
NAS=/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full
mkdir -p "$OUT"

echo "started_at=$(date --iso-8601=seconds)" > "$OUT/stdout.log"
python3 ara/s3-generation/src/s3_private_protocol_transformer_benchmark.py \
  --baseline-summary ara/s3-generation/evidence/s3_private_protocol_battle_full/summary.json \
  --evidence-dir "$OUT" \
  --recipes same_recipe standard_recipe \
  --seeds 71901 71902 71903 \
  --train-samples 30000 --valid-samples 2000 --test-samples 2000 \
  --max-scan 300000 --min-len 8 --max-len 32 \
  --batch-size 64 --num-workers 2 \
  --same-epochs 4 --standard-epochs 8 \
  >> "$OUT/stdout.log" 2> "$OUT/stderr.log"
exit_code=$?

echo "finished_at=$(date --iso-8601=seconds)" >> "$OUT/stdout.log"
echo "exit_code=$exit_code" >> "$OUT/stdout.log"

if [ "$exit_code" -eq 0 ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p "$NAS"
  sudo rsync -a "$OUT/" "$NAS/" || true
  sudo chown -R nio:nio "$NAS" || true
fi

if command -v sendme >/dev/null 2>&1; then
  if [ "$exit_code" -eq 0 ]; then
    sendme -s "TreeHeap Transformer benchmark finished" \
      "io finished S3-PRIVATE-PROTOCOL-TF-C02; evidence: $OUT" || true
  else
    sendme -s "TreeHeap Transformer benchmark failed" \
      "io failed S3-PRIVATE-PROTOCOL-TF-C02 with exit $exit_code; log: $OUT/stderr.log" || true
  fi
fi

exit "$exit_code"
