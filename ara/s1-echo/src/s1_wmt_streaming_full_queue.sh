#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

ROOT_OUT="${1:-ara/s1-echo/evidence/s1_wmt_streaming_full_queue}"
STEPS="${STEPS:-60000}"
LOG_EVERY="${LOG_EVERY:-2000}"
BATCH="${BATCH:-256}"
DIM="${DIM:-128}"
MAX_LEN="${MAX_LEN:-48}"
MODELS="${MODELS:-treeheap,bow,lstm,transformer}"
DATA="${DATA:-/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv}"

mkdir -p "$ROOT_OUT"

run_model() {
  local model="$1"
  local out="$ROOT_OUT/$model"
  rm -rf "$out"
  mkdir -p "$out"
  cat > "$out/command.sh" <<CMD
#!/usr/bin/env bash
set -euo pipefail
cd /home/nio/log/holds/SameTime
python3 ara/s1-echo/src/s1_wmt_streaming_full_smoke.py \\
  --out "$out" \\
  --wmt-path "$DATA" \\
  --models "$model" \\
  --steps "$STEPS" \\
  --log-every "$LOG_EVERY" \\
  --batch "$BATCH" \\
  --max-len "$MAX_LEN" \\
  --dim "$DIM" \\
  --device cuda
CMD
  chmod +x "$out/command.sh"
  {
    echo "model=$model"
    echo "started_at=$(date -Is)"
    free -h
    nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    bash "$out/command.sh"
    echo "finished_at=$(date -Is)"
    free -h
    nvidia-smi --query-gpu=name,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  } > "$out/stdout.log" 2> "$out/stderr.log"
}

{
  echo "queue_started_at=$(date -Is)"
  echo "data=$DATA"
  echo "models=$MODELS"
  echo "steps=$STEPS batch=$BATCH dim=$DIM max_len=$MAX_LEN"
  for model in ${MODELS//,/ }; do
    run_model "$model"
  done
  echo "queue_finished_at=$(date -Is)"
} > "$ROOT_OUT/queue_stdout.log" 2> "$ROOT_OUT/queue_stderr.log"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("ara/s1-echo/evidence/s1_wmt_streaming_full_queue")
rows = []
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text(encoding="utf-8"))
    last = s.get("last_rows", [{}])[-1]
    rows.append({
        "model": p.parent.name,
        "steps": s.get("steps"),
        "batch": s.get("batch"),
        "pilot_pass": s.get("pilot_pass"),
        "last_loss": last.get("loss"),
        "last_align": last.get("align"),
        "last_echo": last.get("echo"),
        "accepted_pairs": last.get("accepted_pairs"),
        "seen_lines": last.get("seen_lines"),
        "elapsed_sec": last.get("elapsed_sec"),
        "cuda_memory_allocated": last.get("cuda_memory_allocated"),
        "cuda_memory_reserved": last.get("cuda_memory_reserved"),
    })
(root / "queue_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
