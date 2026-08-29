#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

runner="ara/s3-generation/src/s3_structural_slot_ownership_d09_scale.py"
root="ara/s3-generation/evidence/s3_structural_slot_ownership_d09_scale"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_slot_ownership_d08/formal_seed10811/subheap/checkpoint_trainable.pt"

test -f "$runner"
test -f "$source_checkpoint"
test -f "$warm_start"
mkdir -p "$root"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_before.csv"

python3 "$runner" --evidence-dir "$root/selftest" --self-test

python3 "$runner" \
  --source-checkpoint "$source_checkpoint" \
  --warm-start "$warm_start" \
  --evidence-dir "$root/formal_seed10901" \
  --seed 10901 \
  --steps 25000 \
  --batch-size 16 \
  --train-rows 200000 \
  --eval-rows 1000 \
  --max-slots 32 \
  --lr 0.0005 \
  --wake-every 2500 \
  --min-delta 0.005 \
  --patience 3 \
  --resume \
  --device cuda \
  2>&1 | tee "$root/formal_seed10901.log"

python3 - "$root/formal_seed10901/summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["claim"] == "S3-STRUCTURAL-SLOT-OWNERSHIP-D09-SCALE"
assert summary["rows"] == {"train": 200000, "valid": 1000, "test": 1000}
assert all(math.isfinite(row["native"]["nll"]) for row in summary["best_causal"].values())
assert summary["reload"]["hash_match"] is True
print(json.dumps({"event": "evidence_validated", "decision": summary["decision"],
                  "gates": summary["gates"], "best_step": summary["best_step"]},
                 ensure_ascii=False))
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/gpu_after.csv"
