#!/bin/bash
set -e
REMOTE_HOST="io.grepcode.cn"
REMOTE_USER="nio"
REMOTE_DIR="/data/homecicd/sametime/code/wmt/experiments_g"

echo "=== 1. Sync code ==="
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}"
scp spr_echo.py spr_tree_layer.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
ssh ${REMOTE_USER}@${REMOTE_HOST} "chmod +x ${REMOTE_DIR}/*.py"

echo "=== 2. Kill lingering containers ==="
# Kill any stale q- containers
ssh ${REMOTE_USER}@${REMOTE_HOST} "docker ps -a --filter name=q- --format '{{.Names}}' | xargs -r docker rm -f 2>/dev/null || true"
ssh ${REMOTE_USER}@${REMOTE_HOST} "pkill -f 'python3 q.py start' 2>/dev/null || true"

echo "=== 3. Clear queue ==="
ssh ${REMOTE_USER}@${REMOTE_HOST} "python3 -c \"import json; json.dump([], open('queue.json','w'))\""

echo "=== 4. Submit all 27 matrix jobs ==="
for depth in 3 5 7; do
  for dim in 64 128 256; do
    for agg in complex_mul simple_add mlp_add; do
      RUN_NAME="echo_L${depth}_D${dim}_${agg}"
      # Notice: NO --rm flag, so q.py can docker wait properly
      CMD="--gpus all --memory=16g --memory-swap=16g -v /data/homecicd/sametime/code/wmt:/workspace -v /mnt/nas/datasets:/mnt/nas/datasets -v /data/homecicd/sametime:/data --mount type=bind,source=/data/datasets/wmt14,target=/data/datasets/wmt14 -w /workspace reg.grepcode.cn/sati/sametime-base:cu121-py310 bash -c 'pip3 install sentencepiece -q 2>/dev/null && python3 -u experiments_g/spr_echo.py --depth ${depth} --dim ${dim} --agg_method ${agg} --run_name ${RUN_NAME}'"
      ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR%experiments_g} && python3 q.py add ${RUN_NAME} \"$CMD\""
    done
  done
done

echo "=== 5. Start q.py with setsid (detached, persistent) ==="
# setsid completely detaches from the SSH session
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR%experiments_g} && setsid python3 q.py start > q_runner.log 2>&1 &"
sleep 2

echo "=== 6. Verify ==="
ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_DIR%experiments_g} && python3 q.py status"
