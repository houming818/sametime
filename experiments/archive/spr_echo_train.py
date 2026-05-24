"""
SPR Echo Train — 纯 numpy 版（不需要 torch）
训练 H 使得分距离 ≈ 嵌入距离
"""
import numpy as np

V, D, d = 128, 4, 8
np.random.seed(42)
tokens = np.random.randn(V, d)
n_leaves = 1 << D

def forward_split(scores, depth=D):
    """递归中位数二分"""
    def split(indices, depth):
        if depth == 0 or len(indices) <= 1:
            return [indices]
        g_scores = scores[indices]
        if len(g_scores) < 2:
            return [indices]
        median = np.median(g_scores)
        left = [idx for idx in indices if scores[idx] <= median]
        right = [idx for idx in indices if scores[idx] > median]
        return split(left, depth-1) + split(right, depth-1)
    return split(list(range(len(scores))), depth)

# ── 训练 ──
H = np.random.randn(d) * 0.5
lr = 0.1

print(f"V={V} d={d} depth={D} leaves={n_leaves}")
print(f"Training H with correlation loss...\n")

for step in range(300):
    scores = tokens @ H                               # (V,)
    score_mean, score_std = scores.mean(), scores.std() + 1e-8
    s_norm = (scores - score_mean) / score_std         # 归一化

    # 嵌入距离矩阵
    diff = tokens[:, None, :] - tokens[None, :, :]    # (V,V,d)
    emb_dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-8)
    emb_mean, emb_std = emb_dist.mean(), emb_dist.std() + 1e-8
    e_norm = (emb_dist - emb_mean) / emb_std

    # 得分差矩阵 + 归一化
    s_dist = np.abs(s_norm[:, None] - s_norm[None, :])
    s_dist_mean, s_dist_std = s_dist.mean(), s_dist.std() + 1e-8
    s_dist_norm = (s_dist - s_dist_mean) / s_dist_std

    e_dist_norm = e_norm
    e_dist_mean, e_dist_std = e_dist_norm.mean(), e_dist_norm.std() + 1e-8
    e_dist_norm = (e_dist_norm - e_dist_mean) / e_dist_std

    loss = ((s_dist_norm - e_dist_norm) ** 2).mean()

    # 梯度: d(loss)/d(H)
    # chain: loss -> s_dist_norm -> s_dist -> s_norm -> scores -> H
    grad_loss_dist = 2 * (s_dist_norm - e_dist_norm) / (V*V)  # (V,V)

    # d(s_dist_norm)/d(s_dist): 1/s_dist_std
    # d(s_dist)/d(s_norm): sign for each pair
    s_diff = s_norm[:, None] - s_norm[None, :]
    grad_sdist_snorm = np.sign(s_diff) / s_dist_std  # (V,V)

    grad_dist = grad_loss_dist * grad_sdist_snorm  # (V,V)

    # d(s_norm)/d(scores) = 1/score_std * (I - 1/V * J)
    # Contributions from each pair to each score
    grad_snorm = np.zeros(V)
    for i in range(V):
        for j in range(V):
            if i != j:
                sign_ij = 1.0 if s_norm[i] > s_norm[j] else -1.0
                grad_snorm[i] += grad_dist[i,j] * (1.0 / s_dist_std) * sign_ij / score_std * (1 - 1/V)
                grad_snorm[i] -= grad_dist[i,j] * (1.0 / s_dist_std) * sign_ij / score_std * (-1/V)

    # d(scores)/d(H) = tokens
    grad_H = tokens.T @ grad_snorm  # (d,)

    H -= lr * grad_H / max(1e-8, np.linalg.norm(grad_H))

    # 监测
    if step % 60 == 0:
        groups = forward_split(scores)
        sizes = [len(g) for g in groups]
        coverage = sum(1 for s in sizes if s > 0)
        print(f"  step {step:3d}: loss={loss:.4f}  "
              f"size_std={np.std(sizes):.1f}  coverage={coverage}/{n_leaves}")

# ── 验证 ──
scores = tokens @ H
groups = forward_split(scores)
sizes = [len(g) for g in groups]

print(f"\n=== Results ===")
print(f"H: {H.round(3).tolist()}")
print(f"Leaf sizes: {sizes}")
print(f"Active: {sum(1 for s in sizes if s > 0)}/{n_leaves}")

# 确定性
scores2 = tokens @ H
groups2 = forward_split(scores2)
same = 0
for i in range(V):
    leaf1 = [j for j, g in enumerate(groups) if i in g][0]
    leaf2 = [j for j, g in enumerate(groups2) if i in g][0]
    same += (leaf1 == leaf2)

print(f"Deterministic: {same}/{V}")
print(f"Mean within-leaf var: {np.mean([tokens[g].var() for g in groups if len(g)>0]):.4f}")

# 对比：随机 H 的组内方差
H_rand = np.random.randn(d)
s_rand = tokens @ H_rand
g_rand = forward_split(s_rand)
var_rand = np.mean([tokens[g].var() for g in g_rand if len(g)>0])
print(f"Random H within-leaf var: {var_rand:.4f}")

if same == V:
    print("\\n=== ECHO TEST PASSED ===")
    # 保存
    H.astype(np.float32).tofile('spr_echo_H.npy')
    print(f"saved H to spr_echo_H.npy")
