#!/usr/bin/env python3
"""
递归砍半做线性回归 — 不用梯度，只用二分排除。

问题：给定 y = 2x + 1 + noise，用树逐层砍半逼近。
对比：梯度下降（拧线）vs 递归砍半（砍空间）
"""
import numpy as np

np.random.seed(42)
N = 1000
x = np.random.uniform(-5, 5, N)
y = 2 * x + 1 + np.random.randn(N) * 1.5

# --- 方法1: 最小二乘（标准线性回归，参考线）---
X = np.stack([x, np.ones_like(x)], axis=1)
w_true = np.linalg.lstsq(X, y, rcond=None)[0]
y_true_line = X @ w_true
print(f"参考线: y = {w_true[0]:.3f}x + {w_true[1]:.3f}")
print(f"MSE: {np.mean((y - y_true_line)**2):.3f}")
print()

# --- 方法2: 递归砍半树 ---
class BisectTree:
    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, x, y, depth=0, max_depth=None):
        n = len(x)
        if max_depth is not None and depth >= max_depth:
            return {"type": "leaf", "value": np.mean(y), "n": n, "depth": depth}
        if n < self.min_samples * 2:
            return {"type": "leaf", "value": np.mean(y), "n": n, "depth": depth}

        # 砍半：在 x 的中位数处分
        split = np.median(x)
        left = x <= split
        right = x > split

        # 如果一侧为空，停止
        if not left.any() or not right.any():
            return {"type": "leaf", "value": np.mean(y), "n": n, "depth": depth}

        return {
            "type": "node",
            "split": split,
            "depth": depth,
            "n": n,
            "left": self.fit(x[left], y[left], depth + 1, max_depth),
            "right": self.fit(x[right], y[right], depth + 1, max_depth),
        }

    def predict_one(self, node, xi):
        if node["type"] == "leaf":
            return node["value"]
        if xi <= node["split"]:
            return self.predict_one(node["left"], xi)
        else:
            return self.predict_one(node["right"], xi)

    def predict(self, tree, xs):
        return np.array([self.predict_one(tree, xi) for xi in xs])

    def count_leaves(self, node):
        if node["type"] == "leaf":
            return 1
        return self.count_leaves(node["left"]) + self.count_leaves(node["right"])


bt = BisectTree(min_samples=5)
tree = bt.fit(x, y, max_depth=None)  # 不限深度

print(f"树总叶子数: {bt.count_leaves(tree)}")
print()

# 对比不同深度的效果
for depth in range(1, 9):
    tree_d = bt.fit(x, y, max_depth=depth)
    y_pred = bt.predict(tree_d, x)
    mse = np.mean((y - y_pred) ** 2)
    leaves = bt.count_leaves(tree_d)
    print(f"深度={depth:2d}  叶子数={leaves:5d}  MSE={mse:.3f}  vs 参考线 MSE={np.mean((y - y_true_line)**2):.3f}")

# --- 对比：用多少个叶子的树能达到梯度下降的精度 ---
print()
print("结论：")
print(f"  梯度下降（拧线）:        2 个参数, MSE={np.mean((y - y_true_line)**2):.3f}")
trees = [(d, bt.fit(x, y, max_depth=d)) for d in range(1, 12)]
for d, t in trees:
    yp = bt.predict(t, x)
    mse = np.mean((y - yp) ** 2)
    leaves = bt.count_leaves(t)
    if mse <= np.mean((y - y_true_line) ** 2) * 1.1:
        print(f"  递归砍半（砍空间）: 深度={d}, {leaves}片叶子, MSE={mse:.3f}")
        break
