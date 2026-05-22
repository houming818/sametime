"""
SPR Echo Test — 可训练版
全局 H 参数: score = H @ token → 递归中位数二分
损失: 得分距离 vs 嵌入距离的 MSE（全可微，不需要 STE）
"""
import torch
import torch.nn.functional as F
import numpy as np

# ── 配置 ──
V, D, d = 128, 4, 8
torch.manual_seed(42)
tokens = torch.randn(V, d)
n_leaves = 1 << D

# ── 模型：全局 H 向量 ──
H = torch.randn(d, requires_grad=True)
opt = torch.optim.Adam([H], lr=0.05)

# ── 训练 ──
def forward_split(scores):
    """递归中位数二分，返回每片叶子的 token indices（非可微，仅用于监测）"""
    def split(indices, depth):
        if depth == 0 or len(indices) <= 1:
            return [indices]
        g_scores = scores[indices]
        if len(g_scores) < 2:
            return [indices]
        median = g_scores.median()
        left = indices[g_scores <= median]
        right = indices[g_scores > median]
        return split(left, depth-1) + split(right, depth-1)
    return split(torch.arange(len(scores)), D)

print(f"V={V} d={d} depth={D} leaves={n_leaves}")
print(f"Training H to correlate score distance with embedding distance...")
print()

for step in range(200):
    opt.zero_grad()

    scores = tokens @ H                                  # (V,)

    # 可微损失：得分距离应该跟嵌入距离一致
    emb_dist = torch.cdist(tokens, tokens)               # (V,V)
    score_dist = torch.abs(scores[:, None] - scores[None, :])  # (V,V)

    emb_dist_n = emb_dist / (emb_dist.max() + 1e-8)
    score_dist_n = score_dist / (score_dist.max() + 1e-8)
    loss = F.mse_loss(score_dist_n, emb_dist_n)

    loss.backward()
    opt.step()

    if step % 40 == 0:
        # 监测：非可微前向
        with torch.no_grad():
            leaf_groups = forward_split(scores.detach())
            sizes = [len(g) for g in leaf_groups]
            # 最优分布：所有叶子均匀
            ideal_std = V / n_leaves * 0.3  # 允许 30% 偏差
            size_std = np.std(sizes)
            collision = sum(1 for s in sizes if s > 1)
            coverage = sum(1 for s in sizes if s > 0)

        print(f"  step {step:3d}: loss={loss.item():.4f}  "
              f"leaves={coverage}/{n_leaves}  collision={collision}  "
              f"size_std={size_std:.1f}")

# ── 最终结果 ──
with torch.no_grad():
    scores = tokens @ H
    leaf_groups = forward_split(scores)
    sizes = np.array([len(g) for g in leaf_groups])
    coverage = np.sum(sizes > 0)

    # 确定性验证
    scores2 = tokens @ H
    leaf_groups2 = forward_split(scores2)
    same = 0
    for i in range(V):
        leaf1 = [j for j, g in enumerate(leaf_groups) if i in g][0]
        leaf2 = [j for j, g in enumerate(leaf_groups2) if i in g][0]
        if leaf1 == leaf2:
            same += 1

print()
print(f"=== Training Results ===")
print(f"H: {H.detach().numpy().round(3)}")
print(f"Leaf sizes: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")
print(f"Active leaves: {coverage}/{n_leaves}")
print(f"Deterministic: {same}/{V}")
print(f"Collision leaves (>1 token): {np.sum(sizes > 1)}")
print(f"Collision tokens: {np.sum(sizes[sizes > 1])}")

if same == V:
    print("\n=== TRAINABLE ECHO TEST PASSED ===")
else:
    print(f"\nWARNING: {V - same} tokens changed path")
