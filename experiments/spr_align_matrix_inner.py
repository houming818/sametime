#!/usr/bin/env python3
"""
Run the full 3×3×3 Translation Alignment matrix in a single Docker container.
Usage inside container: python3 experiments_g/spr_align_matrix_inner.py
"""
import subprocess, itertools, sys, os, json, time

depths = [3, 5, 7]
dims = [64, 128, 256]
aggs = ['complex_mul', 'simple_add', 'mlp_add']

results = []
total = len(depths) * len(dims) * len(aggs)
count = 0
start_all = time.time()

print(f"\n{'='*60}")
print(f"开始运行双语对齐矩阵测试 (共 {total} 组实验)")
print(f"{'='*60}")

for depth, dim, agg in itertools.product(depths, dims, aggs):
    name = f"align_L{depth}_D{dim}_{agg}"
    count += 1
    print(f"\n{'='*60}")
    print(f"[{count}/{total}] {name}")
    print(f"{'='*60}")
    
    t0 = time.time()
    r = subprocess.run([
        "python3", "-u", "experiments_g/spr_trainer.py",
        "--depth", str(depth),
        "--dim", str(dim),
        "--agg_method", agg,
        "--run_name", name,
        "--epochs", "50",
        "--lr", "3e-3"
    ], capture_output=True, text=True)
    elapsed = time.time() - t0
    
    # Print output
    stdout_lines = r.stdout.strip().split("\n")
    if len(stdout_lines) > 10:
        # 只打印前5行和最后5行，避免日志过长
        for line in stdout_lines[:5]:
            print(f"  {line}")
        print("  ...")
        for line in stdout_lines[-5:]:
            print(f"  {line}")
    else:
        for line in stdout_lines:
            print(f"  {line}")
            
    if r.stderr.strip():
        for line in r.stderr.strip().split("\n"):
            print(f"  [ERR] {line}")
    
    print(f"  → Exit: {r.returncode}, Time: {elapsed:.1f}s")
    
    if r.returncode != 0:
        print(f"  ⚠ FAILED (exit={r.returncode})")

total_elapsed = time.time() - start_all
print(f"\n{'='*60}")
print(f"对齐矩阵全部完成！{total} 组实验，总耗时 {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
print(f"{'='*60}")
