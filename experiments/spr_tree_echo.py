"""
SPR Tree Echo — 可训练版
固定二分树 + 每层共享参数 → 训练参数回到全1（echo）
"""
import numpy as np

def tree_forward(tokens, params, depth, level=0):
    if depth == 0 or len(tokens) <= 1:
        return tokens
    p = params[level][:len(tokens)]
    x = tokens * p
    mid = len(x) // 2
    left = tree_forward(x[:mid], params, depth-1, level+1)
    right = tree_forward(x[mid:], params, depth-1, level+1)
    return np.concatenate([left, right])

# ── 训练配置 ──
tokens = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
depth = 3
n_per_level = [len(tokens) // (2**i) for i in range(depth)]  # [8, 4, 2]

# 随机初始化参数（偏离 1）
np.random.seed(42)
params = [np.ones(n) + np.random.randn(n) * 0.5 for n in n_per_level]
initial_params = [p.copy() for p in params]

lr = 0.05
print(f"tokens: {tokens}")
print(f"depth : {depth}  params per level: {n_per_level}")
print(f"initial params: {[p.round(2).tolist() for p in params]}")
print(f"initial loss  : {((tree_forward(tokens, params, depth) - tokens)**2).mean():.4f}")
print()

for step in range(200):
    output = tree_forward(tokens, params, depth)
    loss = ((output - tokens)**2).mean()

    # 有限差分梯度（模型小，直接算）
    eps = 1e-6
    grads = []
    for i, p in enumerate(params):
        grad = np.zeros_like(p)
        for j in range(len(p)):
            old = p[j]
            p[j] = old + eps
            loss_plus = ((tree_forward(tokens, params, depth) - tokens)**2).mean()
            p[j] = old
            grad[j] = (loss_plus - loss) / eps
        grads.append(grad)

    for i in range(len(params)):
        params[i] -= lr * grads[i]

    if step % 40 == 0:
        print(f"step {step:3d}: loss={loss:.6f}")

# ── 结果 ──
final_loss = ((tree_forward(tokens, params, depth) - tokens)**2).mean()
print(f"\nfinal loss  : {final_loss:.6f}")
print(f"final params: {[p.round(3).tolist() for p in params]}")
print(f"target      : {[np.ones(n).tolist() for n in n_per_level]}")
print(f"converged   : {all(np.allclose(p, 1.0, atol=0.01) for p in params)}")
