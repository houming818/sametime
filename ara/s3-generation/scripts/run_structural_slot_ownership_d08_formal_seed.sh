#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

seed="${1:?usage: run_structural_slot_ownership_d08_formal_seed.sh SEED}"
case "$seed" in
  10811|10812|10813) ;;
  *) echo "unregistered D08R1 formal seed: $seed" >&2; exit 2 ;;
esac

runner="ara/s3-generation/src/s3_structural_slot_ownership_d08.py"
root="ara/s3-generation/evidence/s3_structural_slot_ownership_d08"
run="$root/formal_seed${seed}"
checkpoint="ara/s3-generation/evidence/s3_multilevel_read_ablation_c12/formal_seed10101/read/checkpoint_best.pt"

test -f "$runner"
test -f "$checkpoint"
mkdir -p "$root"

power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
python3 - "$power_limit" <<'PY'
import sys
limit = float(sys.argv[1])
assert limit <= 270.5, f"GPU power limit exceeds contract: {limit}W"
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/formal_seed${seed}_gpu_before.csv"

rm -rf "$run"
python3 "$runner" \
  --checkpoint "$checkpoint" \
  --evidence-dir "$run" \
  --seed "$seed" \
  --steps 3000 \
  --batch-size 8 \
  --train-rows 20000 \
  --eval-rows 512 \
  --max-slots 32 \
  --log-every 300 \
  --device cuda \
  2>&1 | tee "$root/formal_seed${seed}.log"

python3 - "$run/summary.json" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

path, seed = Path(sys.argv[1]), int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
assert payload["claim"] == "S3-STRUCTURAL-SLOT-OWNERSHIP-D08R1"
assert payload["config"]["seed"] == seed
assert payload["config"]["steps"] == 3000
assert payload["rows"] == {"train": 20000, "valid": 512, "test": 512}
assert payload["initialization_match"] is True
for arm in payload["arms"].values():
    assert arm["source_sha256_before"] == arm["source_sha256_after"]
    assert arm["language_sha256_before"] == arm["language_sha256_after"]
    for depth in (5, 6, 7):
        row = arm["final_test"][str(depth)]["native"]
        assert math.isfinite(row["nll"])
        assert math.isfinite(row["between_slot_variance"])
print(json.dumps({"event": "formal_seed_validated", "seed": seed,
                  "decision": payload["decision"], "gates": payload["gates"]},
                 ensure_ascii=False))
PY

nvidia-smi --query-gpu=timestamp,name,power.limit,power.draw,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader > "$root/formal_seed${seed}_gpu_after.csv"
