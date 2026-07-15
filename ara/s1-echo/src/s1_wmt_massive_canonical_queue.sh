#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

ROOT_OUT="${1:-ara/s1-echo/evidence/s1_wmt_massive_canonical_scaling_queue}"
mkdir -p "$ROOT_OUT"

DATA_POOL="/mnt/nas/datasets/wmt17/train.zh-en,/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv"
COMMON_ARGS=(
  --wmt-path "$DATA_POOL"
  --models treeheap,bow,lstm,transformer
  --primary-model treeheap
  --baseline-model bow
  --min-len 4
  --max-len 48
  --en-vocab 8192
  --zh-vocab 8192
  --batch 256
  --eval-batch 256
  --max-eval 2000
  --dim 128
  --lr 0.002
  --align-weight 1.0
  --echo-weight 1.0
  --var-weight 1.0
  --target-std 0.05
  --min-random-gain 2.0
  --min-echo-token-acc 0.30
  --device cuda
  --host-label io.grepcode.cn
)

run_one() {
  local name="$1"
  local samples="$2"
  local scan_lines="$3"
  local epochs="$4"
  local out="$ROOT_OUT/$name"
  rm -rf "$out"
  mkdir -p "$out"
  {
    echo "started_at=$(date -Is)"
    echo "name=$name samples=$samples scan_lines=$scan_lines epochs=$epochs"
    nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    python3 ara/s1-echo/src/s1_wmt_canonical_echo_probe.py \
      --out "$out" \
      --samples "$samples" \
      --scan-lines "$scan_lines" \
      --epochs "$epochs" \
      "${COMMON_ARGS[@]}"
    echo "finished_at=$(date -Is)"
    nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  } > "$out/stdout.log" 2> "$out/stderr.log"
}

{
  echo "queue_started_at=$(date -Is)"
  run_one scale_100k 100000 500000 3
  run_one scale_500k 500000 1500000 2
  run_one scale_1m 1000000 3000000 2
  echo "queue_finished_at=$(date -Is)"
} > "$ROOT_OUT/queue_stdout.log" 2> "$ROOT_OUT/queue_stderr.log"

python3 - <<'PY'
import json
from pathlib import Path
root = Path("ara/s1-echo/evidence/s1_wmt_massive_canonical_scaling_queue")
rows = []
for p in sorted(root.glob("scale_*/summary.json")):
    s = json.loads(p.read_text(encoding="utf-8"))
    row = {"run": p.parent.name, "pairs": s["dataset"]["pairs"], "pilot_pass": s["pilot_pass"]}
    for model, payload in s["models"].items():
        m = payload.get("metrics", {}).get("ood")
        if not m:
            continue
        row[f"{model}_margin"] = m["distance_margin_neg_minus_pos"]
        row[f"{model}_retrieval_at_1"] = m["retrieval_at_1"]
        row[f"{model}_retrieval_at_5"] = m["retrieval_at_5"]
        row[f"{model}_entropy"] = m["alignment_entropy"]
    rows.append(row)
(root / "queue_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
