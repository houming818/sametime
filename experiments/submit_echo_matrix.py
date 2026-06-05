#!/usr/bin/env python3
"""Submit all 27 echo matrix jobs to q.py on io.grepcode.cn then start the queue."""
import subprocess, sys, itertools

REMOTE = "nio@io.grepcode.cn"
Q_DIR = "/data/homecicd/sametime/code/wmt"

def ssh(cmd):
    r = subprocess.run(["ssh", REMOTE, cmd], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"SSH ERROR: {r.stderr.strip()}")
    return r.stdout.strip()

# Step 1: clear queue & kill stale
ssh("cd {} && python3 q.py kill 2>/dev/null; python3 -c \"import json; json.dump([], open('queue.json','w'))\"".format(Q_DIR))
ssh("docker ps -a --filter name=q- --format '{{.Names}}' | xargs -r docker rm -f 2>/dev/null || true")
ssh("pkill -f 'python3 q.py start' 2>/dev/null || true")
print("[OK] Queue cleared, stale containers removed.")

# Step 2: add all 27 jobs
depths = [3, 5, 7]
dims = [64, 128, 256]
aggs = ['complex_mul', 'simple_add', 'mlp_add']

count = 0
for depth, dim, agg in itertools.product(depths, dims, aggs):
    name = f"echo_L{depth}_D{dim}_{agg}"
    # Build the CMD string for q.py add
    # Notice: spaces are critical for q.py parsing
    cmd_part = f"--gpus all --memory=16g --memory-swap=16g -v /data/homecicd/sametime/code/wmt:/workspace -v /mnt/nas/datasets:/mnt/nas/datasets -v /data/homecicd/sametime:/data --mount type=bind,source=/data/datasets/wmt14,target=/data/datasets/wmt14 -w /workspace reg.grepcode.cn/sati/sametime-base:cu121-py310 bash -c 'pip3 install sentencepiece -q 2>/dev/null && python3 -u experiments_g/spr_echo.py --depth {} --dim {} --agg_method {} --run_name {}'".format(depth, dim, agg, name)
    # q.py add <name> <rest of command>
    add_cmd = "cd {} && python3 q.py add {} {}".format(Q_DIR, name, cmd_part)
    out = ssh(add_cmd)
    if "added" in out:
        count += 1
        print(f"[ADD] {name}")
    else:
        print(f"[FAIL] {name}: {out}")

print(f"\n[OK] Submitted {count}/27 jobs.")

# Step 3: start q.py persistently
ssh("cd {} && setsid python3 q.py start > q_runner.log 2>&1 &".format(Q_DIR))
import time; time.sleep(2)
status = ssh("cd {} && python3 q.py status".format(Q_DIR))
print(f"\n[STATUS]\n{status}")
