#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

checkpoint="ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/pretrain/checkpoint_best.pt"
runner="ara/s3-generation/src/s3_multilevel_read_ablation_c12.py"
compare="ara/s3-generation/src/s3_multilevel_read_ablation_c12_compare.py"
root="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12r1"

test -f "$checkpoint"
test -f "$runner"
test -f "$compare"
nvidia-smi --query-gpu=power.limit,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader

summary_matches_contract() {
  local summary="$1"
  local arm="$2"
  local seed="$3"
  python3 - "$summary" "$arm" "$seed" <<'PY'
import json
import pathlib
import sys

path, arm, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file():
    raise SystemExit(1)
row = json.loads(path.read_text(encoding="utf-8"))
config = row.get("config", {})
ok = (
    row.get("arm") == arm
    and row.get("mode") == "formal"
    and config.get("seed") == seed
    and config.get("steps") == 25000
    and config.get("batch_size") == 16
    and config.get("train_rows") == 200000
    and config.get("eval_rows") == 1000
    and abs(config.get("lr", 0.0) - 0.002) < 1e-12
    and row.get("rows") == {"train": 200000, "valid": 1000, "test": 1000}
)
raise SystemExit(0 if ok else 2)
PY
}

run_arm() {
  local seed="$1"
  local arm="$2"
  local run_dir="$root/formal_seed${seed}"
  local output="$run_dir/$arm"
  local resume=()

  if summary_matches_contract "$output/summary.json" "$arm" "$seed"; then
    echo "C12-R1 seed $seed arm $arm already complete; skipping"
    return
  fi
  if [[ -f "$output/summary.json" ]]; then
    echo "Existing summary violates C12-R1 contract: $output/summary.json" >&2
    exit 2
  fi
  if [[ -f "$output/checkpoint_progress.pt" ]]; then
    resume=(--resume)
    echo "C12-R1 seed $seed arm $arm resuming"
  fi

  python3 "$runner" \
    --arm "$arm" \
    --mode formal \
    --checkpoint "$checkpoint" \
    --evidence-dir "$output" \
    --seed "$seed" \
    --steps 25000 \
    --batch-size 16 \
    --train-rows 200000 \
    --eval-rows 1000 \
    --lr 0.002 \
    --log-every 500 \
    --max-generation 96 \
    --device cuda \
    "${resume[@]}"
}

for seed in 10102 10103; do
  run_arm "$seed" c10
  run_arm "$seed" read
  run_arm "$seed" read_up
  python3 "$compare" \
    --run-dir "$root/formal_seed${seed}" \
    --output "$root/formal_seed${seed}/comparison.json"
done

echo "C12-R1 two-seed formal queue complete"
