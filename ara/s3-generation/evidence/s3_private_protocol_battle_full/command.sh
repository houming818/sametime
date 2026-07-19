#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

OUT="ara/s3-generation/evidence/s3_private_protocol_battle_full"
NAS="/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_battle_full"
mkdir -p "$OUT" "$NAS"

exec > >(tee -a "$OUT/stdout.log") 2> >(tee -a "$OUT/stderr.log" >&2)

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used,power.draw,power.limit,temperature.gpu --format=csv,noheader

set +e
python3 ara/s3-generation/src/s3_private_protocol_battle.py \
  --evidence-dir "$OUT" \
  --seeds 71901 71902 71903 \
  --variants flat h1 h2 h4 \
  --train-samples 30000 \
  --valid-samples 2000 \
  --test-samples 2000 \
  --max-scan 300000 \
  --epochs 4 \
  --batch-size 64 \
  --dim 192 \
  --hidden 256 \
  --num-workers 2
exit_code=$?
set -e

echo "finished_at=$(date --iso-8601=seconds)"
echo "exit_code=$exit_code"
nvidia-smi --query-gpu=name,memory.total,memory.used,power.draw,power.limit,temperature.gpu --format=csv,noheader || true

rsync -a "$OUT/" "$NAS/"

if command -v sendme >/dev/null 2>&1; then
  if [[ "$exit_code" -eq 0 ]]; then
    sendme -s "TreeHeap private protocol battle complete" \
      "io finished S3-PRIVATE-PROTOCOL-BATTLE-C01; evidence: $OUT" || true
  else
    sendme -s "TreeHeap private protocol battle failed" \
      "io failed S3-PRIVATE-PROTOCOL-BATTLE-C01 with exit $exit_code; log: $OUT/stderr.log" || true
  fi
fi

exit "$exit_code"
