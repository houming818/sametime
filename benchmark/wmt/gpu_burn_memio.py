#!/usr/bin/env python3
"""VRAM Burn — 8h pure memory stress, no monitoring dependencies"""
import torch, time
from datetime import datetime

device = torch.device('cuda')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"VRAM Burn 8h | {torch.cuda.get_device_name(0)}")
log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")

N = 8192
tensors = [torch.randn(N, N, device=device) for _ in range(3)]
torch.cuda.synchronize()
log(f"3x{N} tensors allocated, starting burn...")

start = time.time()
i = 0
while time.time() - start < 8*3600:
    try:
        for t in tensors:
            t2 = t.clone()       # VRAM read + write
            t3 = t2 * 1.01       # element-wise (memory-bound)
            del t2, t3
        x = tensors[0][:1024,:1024].clone()
        del x
        i += 1
    except RuntimeError as e:
        log(f"FATAL CUDA at iter={i}: {e}")
        break
    except Exception as e:
        log(f"FATAL at iter={i}: {e}")
        break
    
    if i % 500 == 0:
        elapsed = time.time()-start
        used = torch.cuda.memory_allocated()/1024**3
        log(f"i={i:6d} {elapsed/3600:.1f}h VRAM:{used:.1f}GB")

elapsed = time.time()-start
log(f"Done: {elapsed/3600:.1f}h, {i} iters")
