# submit_echo_matrix_on_io.sh — run this ON io.grepcode.cn
set -e
Q_DIR="/data/homecicd/sametime/code/wmt"
cd "$Q_DIR"

echo "=== Kill stale ==="
python3 q.py kill 2>/dev/null || true
docker ps -a --filter name=q- --format '{{.Names}}' | xargs -r docker rm -f 2>/dev/null || true
pkill -f 'python3 q.py start' 2>/dev/null || true

echo "=== Clear queue ==="
python3 -c "import json; json.dump([], open('queue.json','w'))"

echo "=== Add 27 jobs ==="
for depth in 3 5 7; do
  for dim in 64 128 256; do
    for agg in complex_mul simple_add mlp_add; do
      NAME="echo_L${depth}_D${dim}_${agg}"
      CMD="--gpus all --memory=16g --memory-swap=16g \
-v /data/homecicd/sametime/code/wmt:/workspace \
-v /mnt/nas/datasets:/mnt/nas/datasets \
-v /data/homecicd/sametime:/data \
--mount type=bind,source=/data/datasets/wmt14,target=/data/datasets/wmt14 \
-w /workspace \
reg.grepcode.cn/sati/sametime-base:cu121-py310 \
bash -c 'pip3 install sentencepiece -q 2>/dev/null && python3 -u experiments_g/spr_echo.py --depth ${depth} --dim ${dim} --agg_method ${agg} --run_name ${NAME}'"
      python3 q.py add "$NAME" $CMD
    done
  done
done

echo "=== Start queue in background ==="
nohup python3 q.py start > q_runner.log 2>&1 &
sleep 2
python3 q.py status
echo "=== DONE ==="
