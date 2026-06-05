#!/usr/bin/env python3
"""Remote submitter: run this ON io.grepcode.cn to submit all 27 echo matrix jobs."""
import json, subprocess, sys, os, itertools, time

Q_DIR = "/data/homecicd/sametime/code/wmt"
os.chdir(Q_DIR)

# 1. Clear
subprocess.run(["python3", "q.py", "kill"], capture_output=True)
subprocess.run(["docker", "ps", "-a", "--filter", "name=q-", "--format", "{{.Names}}"], 
               capture_output=True, text=True)
json.dump([], open("queue.json","w"))
print("[OK] Queue cleared")

# 2. Add all 27 jobs
depths = [3, 5, 7]
dims = [64, 128, 256]
aggs = ['complex_mul', 'simple_add', 'mlp_add']

cnt = 0
for depth, dim, agg in itertools.product(depths, dims, aggs):
    name = f"echo_L{depth}_D{dim}_{agg}"
    script = f"pip3 install sentencepiece -q 2>/dev/null && python3 -u experiments_g/spr_echo.py --depth {depth} --dim {dim} --agg_method {agg} --run_name {name}"
    cmd_part = (
        f"--gpus all --memory=16g --memory-swap=16g "
        f"-v /data/homecicd/sametime/code/wmt:/workspace "
        f"-v /mnt/nas/datasets:/mnt/nas/datasets "
        f"-v /data/homecicd/sametime:/data "
        f"--mount type=bind,source=/data/datasets/wmt14,target=/data/datasets/wmt14 "
        f"-w /workspace "
        f"reg.grepcode.cn/sati/sametime-base:cu121-py310 "
        f"bash -c '{script}'"
    )
    r = subprocess.run(["python3", "q.py", "add", name, cmd_part], 
                       capture_output=True, text=True)
    if "added" in r.stdout:
        cnt += 1
    else:
        print(f"FAIL add {name}: {r.stdout.strip()} {r.stderr.strip()}")
print(f"[OK] Submitted {cnt}/27")

# 3. Start (detached)
proc = subprocess.Popen(
    ["setsid", "python3", "q.py", "start"],
    stdout=open("q_runner.log", "w"),
    stderr=subprocess.STDOUT,
    preexec_fn=os.setpgrp
)
time.sleep(2)
status = subprocess.run(["python3", "q.py", "status"], capture_output=True, text=True)
print(f"[STATUS]\n{status.stdout}")
