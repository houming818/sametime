#!/usr/bin/env bash
set -euo pipefail

cd /home/nio/log/holds/SameTime

cli="ara/s3-generation/src/treeheap_epoch_translate_cli.py"
root="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/cli_examples_20260907"
base="ara/s3-generation/evidence/s3_epoch_repeat_scaling_e01/formal_seed11301"
mkdir -p "$root"

for arm in treeheap-63m treeheap-106m; do
  checkpoint="$base/$arm/checkpoint_latest.pt"
  test -f "$checkpoint"
  python3 "$cli" --checkpoint "$checkpoint" --direction en2zh --depth all \
    --text "The child put the red book on the table." \
    --text "Although it was raining, we decided to walk home." \
    --text "Sisyphus is pushing the stone up the mountain." \
    > "$root/${arm}_en2zh.json"
  python3 "$cli" --checkpoint "$checkpoint" --direction zh2en --depth all \
    --text "孩子把红色的书放在桌上。" \
    --text "虽然下着雨，我们还是决定步行回家。" \
    --text "西西弗斯正在把石头推上山。" \
    > "$root/${arm}_zh2en.json"
done

python3 - "$root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("treeheap-*.json")):
    payload = json.load(open(path, encoding="utf-8"))
    assert len(payload["results"]) == 9
    rows.extend(payload["results"])
json.dump({"count": len(rows), "results": rows}, open(root / "all_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"event": "cli_examples_complete", "count": len(rows)}, ensure_ascii=False))
PY
