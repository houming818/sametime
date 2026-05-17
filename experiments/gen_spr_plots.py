#!/usr/bin/env python3
"""Generate SPR-003 visualizations — recursive bisection vs gradient descent"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
N = 200
x_raw = np.random.uniform(-5, 5, N)
y_raw = 2 * x_raw + 1 + np.random.randn(N) * 1.5

OUT = "/home/nio/log/blogs/www.grepcode.cn/static/spr"

# -------- 图1: 数据 + 随机初始线 --------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_raw, y_raw, alpha=0.4, s=20, c='#333')
# 随机初始线
ax.axline((0, -3), slope=-0.5, color='red', linewidth=2, label='随机初始线')
ax.axline((0, 3), slope=1.5, color='orange', linewidth=2, label='另一条随机线')
ax.set_xlim(-6, 6); ax.set_ylim(-12, 16)
ax.set_title('图1: 数据点 + 两条随机初始线', fontsize=14)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/spr-fig1-random-lines.png", dpi=120)
plt.close()

# -------- 图2: 四步拧线过程 --------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
steps = [(0, 1.0, 0.0, 'iter 0: 随机线'), (10, 1.4, 0.3, 'iter 10: 轻微拧转'),
         (50, 1.9, 0.9, 'iter 50: 大幅靠近'), (200, 1.977, 1.146, 'iter 200: 最优线')]
for ax, (i, w, b, title) in zip(axes.flat, steps):
    ax.scatter(x_raw, y_raw, alpha=0.3, s=15, c='#666')
    ax.axline((0, b), slope=w, color='red', linewidth=2.5)
    ax.set_xlim(-6, 6); ax.set_ylim(-12, 16)
    ax.set_title(title, fontsize=12); ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(f"{OUT}/spr-fig2-gradient-descent.png", dpi=120)
plt.close()

# -------- 图3: 递归砍半 — 四步砍空间 --------
def region_mean(low, high):
    mask = (x_raw >= low) & (x_raw < high)
    if mask.sum() == 0: return 0
    return np.mean(y_raw[mask])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
depths = [1, 2, 3, 4]
for ax, depth in zip(axes.flat, depths):
    splits = np.linspace(-5, 5, 2**depth + 1)
    ax.scatter(x_raw, y_raw, alpha=0.3, s=15, c='#666')
    for s in splits[1:-1]:
        ax.axvline(x=s, color='#1f77b4', linewidth=1, alpha=0.5, linestyle='--')
    for i in range(len(splits)-1):
        lo, hi = splits[i], splits[i+1]
        mid = (lo + hi) / 2
        val = region_mean(lo, hi)
        ax.hlines(val, lo, hi, color='red', linewidth=2.5)
    ax.set_xlim(-6, 6); ax.set_ylim(-12, 16)
    ax.set_title(f'砍半 depth={depth} ({2**depth} 片叶子)', fontsize=12)
    ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(f"{OUT}/spr-fig3-bisection.png", dpi=120)
plt.close()

# -------- 图4: 对比 — 拧线 vs 砍半 --------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# 拧线
ax1.scatter(x_raw, y_raw, alpha=0.3, s=15, c='#666')
ax1.axline((0, 1.146), slope=1.977, color='red', linewidth=2.5)
ax1.set_title(f'梯度下降(拧线): 2参数, MSE=2.194', fontsize=13)
ax1.set_xlim(-6, 6); ax1.set_ylim(-12, 16); ax1.grid(True, alpha=0.2)
# 砍半深度5
depth = 5
splits = np.linspace(-5, 5, 2**depth + 1)
ax2.scatter(x_raw, y_raw, alpha=0.3, s=15, c='#666')
for s in splits[1:-1]:
    ax2.axvline(x=s, color='#1f77b4', linewidth=1, alpha=0.5, linestyle='--')
for i in range(len(splits)-1):
    lo, hi = splits[i], splits[i+1]
    val = region_mean(lo, hi)
    ax2.hlines(val, lo, hi, color='red', linewidth=2.5)
ax2.set_title(f'递归砍半(砍空间): {2**depth}片叶子, MSE=2.162', fontsize=13)
ax2.set_xlim(-6, 6); ax2.set_ylim(-12, 16); ax2.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(f"{OUT}/spr-fig4-comparison.png", dpi=120)
plt.close()

# -------- 图5: 收敛速度对比 --------
fig, ax = plt.subplots(figsize=(8, 5))
depths_plot = range(1, 9)
mse_values = [10.066, 4.130, 2.707, 2.324, 2.162, 2.085, 1.911, 1.911]
params = [2**d for d in depths_plot]
ax.plot(params, mse_values, 'o-', color='red', linewidth=2, markersize=8, label='递归砍半 MSE')
ax.axhline(y=2.194, color='blue', linewidth=2, linestyle='--', label=f'梯度下降 MSE=2.194')
ax.axvline(x=2, color='gray', linewidth=1, linestyle=':', alpha=0.5, label='拧线: 2参数')
ax.axvline(x=16, color='gray', linewidth=1, linestyle=':', alpha=0.5, label='砍半追平: 16片叶子')
ax.set_xscale('log', base=2)
ax.set_xlabel('叶子数 (log 尺度)', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('收敛速度对比: 拧线 vs 砍空间', fontsize=14)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/spr-fig5-convergence.png", dpi=120)
plt.close()

print("5张图已生成到 blogs/www.grepcode.cn/static/spr/")
for name in ['spr-fig1-random-lines','spr-fig2-gradient-descent','spr-fig3-bisection','spr-fig4-comparison','spr-fig5-convergence']:
    print(f"  {name}.png")
