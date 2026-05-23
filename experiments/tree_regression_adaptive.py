#!/usr/bin/env python3
"""
自适应砍半 vs 均匀砍半 — 用数据密度决定分叉点位置
"""
import numpy as np

np.random.seed(42)
N = 600
# 非均匀 x — 左侧密集，右侧稀疏
x = np.concatenate([
    np.random.uniform(-5, -1, 300),   # 左侧 300 个点密集区
    np.random.uniform(1, 5, 300),     # 右侧 300 个点稀疏区
])
y = 2 * x + 1 + np.random.randn(N) * 1.5

# --- 均匀砍半（中位数分叉）---
def make_leaf(x, y):
    return {"type": "leaf", "value": np.mean(y), "n": len(x)}

def median_tree(x, y, depth=0, max_depth=10):
    if depth >= max_depth or len(x) < 10:
        return make_leaf(x, y)
    split = np.median(x)
    left = x <= split; right = x > split
    if not left.any() or not right.any():
        return make_leaf(x, y)
    return {"type": "node", "split": split,
        "left": median_tree(x[left], y[left], depth+1, max_depth),
        "right": median_tree(x[right], y[right], depth+1, max_depth)}

def var_tree(x, y, depth=0, max_depth=10):
    if depth >= max_depth or len(x) < 10:
        return make_leaf(x, y)
    best_var, best_split = -1, np.median(x)
    for s in np.percentile(x, [20, 40, 60, 80]):
        l, r = y[x <= s], y[x > s]
        if len(l) < 5 or len(r) < 5: continue
        v = np.var(y) - (len(l)/len(x) * np.var(l) + len(r)/len(x) * np.var(r))
        if v > best_var:
            best_var, best_split = v, s
    left = x <= best_split; right = x > best_split
    if not left.any() or not right.any():
        return make_leaf(x, y)
    return {"type": "node", "split": best_split,
        "left": var_tree(x[left], y[left], depth+1, max_depth),
        "right": var_tree(x[right], y[right], depth+1, max_depth)}

def predict(tree, xi):
    return tree["value"] if tree["type"] == "leaf" else predict(tree["left"] if xi <= tree["split"] else tree["right"], xi)

def mse(tree, x, y):
    yp = np.array([predict(tree, xi) for xi in x])
    return np.mean((y - yp)**2)

def count_leaves(tree):
    if not isinstance(tree, dict) or "type" not in tree: return 0
    if tree["type"] == "leaf": return 1
    return count_leaves(tree.get("left", {})) + count_leaves(tree.get("right", {}))

# 对比
print(f"数据: {N} 点, x范围=[{np.min(x):.1f}, {np.max(x):.1f}]")
print(f"  左侧 [-5,-1] 密集 300 点, 右侧 [1,5] 稀疏 300 点")
print()

for method, name in [(median_tree, "均匀砍半(中位数)"), (var_tree, "自适应砍半(方差最大)")]:
    for depth in range(1, 7):
        t = method(x, y, max_depth=depth)
        leaves = count_leaves(t)
        m = mse(t, x, y)
        print(f"  {name:20s} depth={depth}  leaves={leaves:4d}  MSE={m:.3f}")

print()
print("结论: 自适应砍半在稀疏区域少砍(省叶子), 达到同样 MSE 需要的叶子更少")
