#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

claim_root="ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11"
formal="$claim_root/formal_seed11101"
smoke="$claim_root/smoke_seed11101"
runner="ara/s3-generation/src/s3_structural_protocol_capacity_ladder_d11.py"
compare="ara/s3-generation/src/s3_structural_protocol_capacity_ladder_d11_r1_compare.py"
cpu_test="ara/s3-generation/src/test_structural_protocol_capacity_ladder_d11.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
d10_root="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001"
warm_start="$d10_root/task/checkpoint_best.pt"

test -f "$runner"
test -f "$compare"
test -f "$cpu_test"
test -f "$source_checkpoint"
test -f "$warm_start"
test -f "$smoke/comparison.json"
test -f "$formal/treeheap-63m/summary.json"

python3 "$cpu_test"

python3 - "$smoke/comparison.json" "$formal/treeheap-63m/summary.json" <<'PY'
import json, sys
smoke = json.load(open(sys.argv[1], encoding="utf-8"))
base = json.load(open(sys.argv[2], encoding="utf-8"))
assert smoke["decision"] == "first_scale_rung_supported"
assert smoke["metrics"]["initial_nll_delta"] <= 1e-8
assert base["step"] == 25_000 and base["cursor"] == 400_488
for gate in (
    "steps_complete", "finite", "input_causality", "structure",
    "source_frozen", "trainable_updated", "reload",
):
    assert base["gates"][gate], f"baseline safety gate failed: {gate}"
PY

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
assert float(sys.argv[1]) <= 270.5
PY

mkdir -p "$formal"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$formal/gpu_r1_before.csv"

set +e
python3 "$runner" --mode formal --arm treeheap-106m --extra-dim 384 \
  --max-steps 25000 --wake-every 5000 \
  --evidence-dir "$formal/treeheap-106m" \
  --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
  --seed 11101 --ownership-seed 11102 --batch-size 16 --device cuda \
  2>&1 | tee "$formal/treeheap-106m-r1.log"
runner_rc=${PIPESTATUS[0]}
set -e

# Exit 3 means the fixed-budget run completed but a product-quality observation failed.
# It is not a training interruption. All safety gates are checked independently below.
if [[ "$runner_rc" -ne 0 && "$runner_rc" -ne 3 ]]; then
  exit "$runner_rc"
fi

python3 - "$formal/treeheap-106m/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
for gate in (
    "steps_complete", "finite", "input_causality", "structure",
    "source_frozen", "trainable_updated", "reload",
):
    assert x["gates"][gate], f"scale safety gate failed: {gate}"
PY

python3 "$compare" --root "$formal" | tee "$formal/compare_r1.log"
nvidia-smi --query-gpu=timestamp,name,pstate,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$formal/gpu_r1_after.csv"

sendme -s "D11-R1 fixed-budget 63M vs 106M capacity rung finished" \
  -f "$formal/comparison_r1.json" || true
