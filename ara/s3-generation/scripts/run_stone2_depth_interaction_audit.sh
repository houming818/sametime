#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime
OUT="ara/s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/depth_interaction_audit.json"
CKPT="ara/s3-generation/evidence/s3_stone2_integrated_c03/smoke_seed16101/task/PT/checkpoint_best.pt"
mkdir -p "$(dirname "$OUT")"

finish() {
  status=$?
  if [[ $status -eq 0 ]]; then
    summary=$(python3 - "$OUT" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"full_value":x["full_value"],"shapley":x["shapley"],"interpretation":x["interpretation"]},ensure_ascii=False))
PY
)
    sendme "STONE-2 C03 深度组合诊断完成" "$summary" || true
  else
    sendme "STONE-2 C03 深度组合诊断失败" "exit=$status; output=$OUT" || true
  fi
  exit "$status"
}
trap finish EXIT

python3 ara/s3-generation/src/s3_stone2_depth_interaction_audit.py \
  --checkpoint "$CKPT" \
  --output "$OUT" \
  --device cuda \
  --eval-rows 256
