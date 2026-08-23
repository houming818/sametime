#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

SOURCE_NONPAR="ara/data-quality/evidence/full_nonparallel_cleaning/formal_seed15106"
SOURCE_PAR="ara/data-quality/evidence/full_parallel_cleaning/formal_14m_v1"
DATA_ROOT="/home/nio/datasets/nio/releases"
REGISTRY="ara/data-quality/datasets/releases"

verify_or_run() {
  local output="$1"
  local manifest="$2"
  shift 2
  if [[ -e "$output" || -e "$manifest" ]]; then
    if [[ ! -f "$output" || ! -f "$manifest" ]]; then
      echo "partial immutable release: $output / $manifest" >&2
      return 1
    fi
    python3 - "$output" "$manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, manifest = map(Path, sys.argv[1:])
x=json.loads(manifest.read_text(encoding="utf-8"))
h=hashlib.sha256()
with data.open("rb") as f:
    for block in iter(lambda:f.read(8*1024*1024), b""):
        h.update(block)
if h.hexdigest() != x["output_sha256"]:
    raise SystemExit("existing release hash mismatch")
print(json.dumps({"event":"release_reused","dataset":x.get("dataset_id"),"rows":x["rows"]},ensure_ascii=False))
PY
  else
    "$@"
  fi
}

materialize_nonparallel() {
  local family="$1"
  local name="$2"
  local threshold="$3"
  local data_dir="$DATA_ROOT/$name"
  local registry_dir="$REGISTRY/$name"
  mkdir -p "$data_dir" "$registry_dir"
  local args=(
    python3 ara/data-quality/src/materialize_nonparallel_view.py
    --input "$SOURCE_NONPAR/$family"
    --output "$data_dir/data.jsonl"
    --manifest "$registry_dir/materialized_view.json"
    --dataset-id "$name"
    --family "$family"
  )
  if [[ -n "$threshold" ]]; then
    args+=(--threshold "$threshold")
  fi
  verify_or_run "$data_dir/data.jsonl" "$registry_dir/materialized_view.json" "${args[@]}"
}

finish() {
  status=$?
  if [[ $status -eq 0 ]]; then
    summary=$(python3 - <<'PY'
import json
from pathlib import Path
root=Path("ara/data-quality/datasets/releases")
names=[
 "NioText-ZH-Integrity-2985K-v1",
 "NioClean-ZHEN-S098-7M-v2",
 "NioQA-ZH-S090-v1",
 "NioQA-ZH-S095-v1",
 "NioQA-ZH-S098-v1",
]
print(json.dumps({n:json.loads((root/n/"materialized_view.json").read_text())["rows"] for n in names},ensure_ascii=False))
PY
)
    sendme "STONE-2 核心数据 release 已固化" "$summary" || true
  else
    sendme "STONE-2 核心数据 release 固化失败" "exit=$status" || true
  fi
  exit "$status"
}
trap finish EXIT

materialize_nonparallel mono NioText-ZH-Integrity-2985K-v1 ""
materialize_nonparallel qa NioQA-ZH-S090-v1 0.90
materialize_nonparallel qa NioQA-ZH-S095-v1 0.95
materialize_nonparallel qa NioQA-ZH-S098-v1 0.98

PAR_NAME="NioClean-ZHEN-S098-7M-v2"
PAR_DATA="$DATA_ROOT/$PAR_NAME/pairs.tsv"
PAR_MANIFEST="$REGISTRY/$PAR_NAME/materialized_view.json"
mkdir -p "$(dirname "$PAR_DATA")" "$(dirname "$PAR_MANIFEST")"
verify_or_run "$PAR_DATA" "$PAR_MANIFEST" \
  python3 ara/data-quality/src/materialize_scored_parallel_view.py \
    --input "$SOURCE_PAR" \
    --output "$PAR_DATA" \
    --manifest "$PAR_MANIFEST" \
    --threshold 0.98 \
    --expected-rows 7304358 \
    --expected-shards 57

python3 - <<'PY'
import hashlib, json
from pathlib import Path

root=Path("ara/data-quality/datasets/releases")
names=[
 "NioText-ZH-Integrity-2985K-v1",
 "NioClean-ZHEN-S098-7M-v2",
 "NioQA-ZH-S090-v1",
 "NioQA-ZH-S095-v1",
 "NioQA-ZH-S098-v1",
]
members=[]
for name in names:
    path=root/name/"materialized_view.json"
    raw=path.read_bytes()
    item=json.loads(raw)
    members.append({
        "dataset_id":name,
        "rows":item["rows"],
        "manifest":str(path),
        "manifest_sha256":hashlib.sha256(raw).hexdigest(),
        "data_sha256":item["output_sha256"],
    })
payload={"schema":"nio.stone2-core-data.v1","members":members}
canonical=json.dumps(payload,ensure_ascii=False,separators=(",",":"),sort_keys=True).encode()
payload["root_sha256"]=hashlib.sha256(canonical).hexdigest()
out=root/"STONE2-CORE-v1.json"
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False))
PY
