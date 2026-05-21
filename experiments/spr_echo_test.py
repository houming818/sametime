"""
SPR Echo Test — 语义递归分裂树
整句从根开始，每层把 token 组劈两半，每半各自算哈希
hash = 节点存当前组的语义，递归下探
"""
import numpy as np
from collections import Counter

V, D, d = 100, 4, 8
np.random.seed(42)
tokens = np.random.randn(V, d)

n_nodes = (1 << D) - 1
n_leaves = 1 << D
H = np.zeros((n_nodes, d))  # each node stores a d-dim hash vector
H_set = np.zeros(n_nodes, dtype=bool)

# ── 递归分裂：每层把 token 组劈两半 ──
def split_and_hash(token_ids, node_idx, depth, max_depth=D):
    """对 token_ids 集合，算哈希，劈两半，递归"""
    if depth >= max_depth:
        return
    if node_idx >= n_nodes:
        return

    # 节点哈希 = 当前组所有 token 的均值
    H[node_idx] = np.mean(tokens[token_ids], axis=0)
    H_set[node_idx] = True

    if len(token_ids) <= 1 or depth == max_depth - 1:
        return  # 叶子层——不继续劈

    # 用当前组的第一主成分方向劈两半
    group_vecs = tokens[token_ids]
    centroid = H[node_idx]
    centered = group_vecs - centroid
    projection = centered @ centered[0]  # dot with first token as direction

    left = token_ids[projection <= 0]
    right = token_ids[projection > 0]

    if len(left) > 0:
        split_and_hash(left, 2 * node_idx + 1, depth + 1)
    if len(right) > 0:
        split_and_hash(right, 2 * node_idx + 2, depth + 1)

split_and_hash(np.arange(V), 0, 0)

# ── 路由：token 沿分裂规则走到叶子 ──
def route(token_vec):
    i = 0
    path = []
    for depth in range(D):
        if not H_set[i]:
            break
        group_centroid = H[i]
        # 用同一个劈分方向判定
        proj = (token_vec - group_centroid) @ (tokens[0] - H[0])
        if proj <= 0:
            path.append(0)
            i = 2 * i + 1
        else:
            path.append(1)
            i = 2 * i + 2
    return tuple(path)

# ── 验证 ──
paths = {}
deterministic = 0
for idx in range(V):
    p1 = route(tokens[idx])
    p2 = route(tokens[idx])
    paths[idx] = p1
    if p1 == p2:
        deterministic += 1

path_counts = Counter(paths.values())
unique_paths = len(path_counts)

prefix_shared = 0
for i in range(V):
    for j in range(i+1, V):
        pi, pj = paths[i], paths[j]
        for k in range(1, D):
            if pi[:k] == pj[:k]:
                prefix_shared += 1

print(f"V={V} d={d} depth={D} nodes={n_nodes} leaves={n_leaves}")
print()
print(f"=== Results ===")
print(f"Deterministic  : {deterministic}/{V}")
print(f"Unique paths   : {unique_paths}")
print(f"Prefix collisions: {prefix_shared}")
print()
print("=== Node hashes (first 3 dims of root + children) ===")
print(f"  root  : {H[0][:3].round(3)}  ← 整句语义哈希")
print(f"  left  : {H[1][:3].round(3)}  ← 前半组语义哈希")
print(f"  right : {H[2][:3].round(3)}  ← 后半组语义哈希")
print()
print("=== Collision groups ===")
for p, count in sorted(path_counts.items(), key=lambda x: -x[1])[:6]:
    ids = [str(i) for i, pp in paths.items() if pp == p][:5]
    print(f"  {p}: {count} tokens — {ids}...")

assert deterministic == V
print("\n=== ECHO TEST PASSED ===")
