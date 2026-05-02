#!/usr/bin/env python3
"""
GPU Burn-In Test — 8 hour stability check for RTX 3090
Monitors: temp, power, clock every 30s. Exits clean or on OOM.
"""

import torch
import time
import subprocess
import sys
from datetime import datetime

DURATION = 8 * 3600  # 8 hours
LOG_INTERVAL = 30     # seconds

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def gpu_stats():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=temperature.gpu,power.draw,clocks.current.sm,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], text=True).strip()
        return out
    except:
        return "GPU_DOWN"

log("===== GPU Burn-In Started =====")
log(f"Duration: {DURATION}s ({DURATION/3600:.1f}h)")
log(f"CUDA: {torch.cuda.is_available()}")
log(f"Device: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda')

# Allocate large tensors to stress VRAM
size = 8192
log(f"Allocating {size}x{size} float32 tensors ({size*size*4/1024/1024:.0f} MB each)...")

# Warm up
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
torch.cuda.synchronize()
log("Warmup done, starting burn loop...")

start = time.time()
iteration = 0
total_ops = 0

try:
    while time.time() - start < DURATION:
        c = torch.mm(a, b)
        d = torch.mm(c, a)
        e = torch.mm(d, b)
        torch.cuda.synchronize()
        iteration += 1
        total_ops += 3
        
        if iteration % 50 == 0:
            stats = gpu_stats()
            elapsed = time.time() - start
            remaining = DURATION - elapsed
            ops_per_iter = total_ops * size * size * size * 2  # FLOP approximation
            log(f"iter={iteration:6d} | elapsed={elapsed/3600:.1f}h | remain={remaining/3600:.1f}h | GPU: {stats}")
            
            if stats == "GPU_DOWN":
                log("!!! GPU DISAPPEARED — Xid 79 detected !!!")
                break

except KeyboardInterrupt:
    log("Stopped by user")
except Exception as e:
    log(f"FATAL: {e}")
    log(f"Last GPU state: {gpu_stats()}")

elapsed = time.time() - start
log(f"===== Completed: {elapsed/3600:.1f}h, {iteration} iterations =====")
