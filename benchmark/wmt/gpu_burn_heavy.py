#!/usr/bin/env python3
"""
GPU Heavy Burn — 390W stress test for RTX 3090 ROG Strix
Large tensors + mixed ops to push power draw to 370-390W
"""

import torch
import subprocess
import time
from datetime import datetime

DURATION = 8 * 3600
LOG_INTERVAL = 30

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def stats():
    try:
        return subprocess.check_output([
            'nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.current.sm,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
    except:
        return "GPU_DOWN"

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')
log(f"===== Heavy Burn: 390W, {DURATION/3600:.0f}h =====")
log(f"Device: {torch.cuda.get_device_name(0)}")

# Two large tensor pairs for alternating stress
S = 10240
a = torch.randn(S, S, device=device)
b = torch.randn(S, S, device=device)
x = torch.randn(S, S, device=device)
y = torch.randn(S, S, device=device)
torch.cuda.synchronize()

start = time.time()
i = 0

try:
    while time.time() - start < DURATION:
        c = torch.mm(a, b)
        d = torch.mm(c, x)
        e = torch.mm(d, y)
        f = torch.mm(a, y)
        g = torch.mm(f, b)
        h = torch.mm(g, x)
        torch.cuda.synchronize()
        i += 1
        
        if i % 30 == 0:
            st = stats()
            elapsed = time.time() - start
            log(f"iter={i:5d}  elapsed={elapsed/3600:.1f}h  remain={(DURATION-elapsed)/3600:.1f}h  GPU: {st}")
            if st == "GPU_DOWN":
                log("!!! Xid 79 !!!")
                break

except Exception as e:
    log(f"FATAL: {e}\nLast: {stats()}")

elapsed = time.time() - start
log(f"===== Done: {elapsed/3600:.1f}h, {i} iters =====")
