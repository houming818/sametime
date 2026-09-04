#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

root="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/smoke_seed11301"
runner="ara/s3-generation/src/s3_epoch_repeat_scaling.py"
test_runner="ara/s3-generation/src/test_epoch_repeat_scaling.py"
source_checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"
warm_start="ara/s3-generation/evidence/s3_structural_protocol_full_pipeline_d10_recovery_r1/formal_seed11001/task/checkpoint_best.pt"

python3 "$test_runner"
mkdir -p "$root"

for spec in "treeheap-63m:0" "treeheap-106m:384"; do
  arm="${spec%%:*}"
  extra_dim="${spec##*:}"
  resume="ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11/formal_seed11101/$arm/checkpoint_latest.pt"
  test -f "$resume"
  python3 "$runner" \
    --mode smoke --arm "$arm" --extra-dim "$extra_dim" \
    --source-checkpoint "$source_checkpoint" --warm-start "$warm_start" \
    --resume-checkpoint "$resume" --evidence-dir "$root/$arm" \
    --batch-size 64 --max-segment-steps 20 --target-cursor 405608 \
    --expected-start-cursor 400488 --corpus-pass 2 --direction-flip 0 \
    --seed 11101 --ownership-seed 11102 --device cuda \
    2>&1 | tee "$root/$arm.log"
done

python3 - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = {}
for arm in ("treeheap-63m", "treeheap-106m"):
    x = json.load(open(root / arm / "summary.json", encoding="utf-8"))
    assert all(x["gates"].values()), (arm, x["gates"])
    assert x["step"] == 25020
    assert x["cursor"] == 401768
    assert not x["completed"]
    rows[arm] = x
assert rows["treeheap-63m"]["cursor"] == rows["treeheap-106m"]["cursor"]
print(json.dumps({"claim": "S3-EPOCH-REPEAT-SCALING-E01", "smoke": "passed"}, ensure_ascii=False))
PY

sendme -s "Epoch Repeat Scaling real-checkpoint smoke completed" \
  -f "$root/treeheap-106m/summary.json" || true

