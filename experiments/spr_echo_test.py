"""
SPR Echo Test — 路径即哈希版本
token → 每层判定 → 路径 = 哈希 → echo = 同路径
碰撞 = 两个 token 哈希前缀相同
"""
import numpy as np
from collections import Counter

V, D, d = 100, 4, 8          # more depth for richer collision analysis
np.random.seed(42)
tokens = np.random.randn(V, d)

n_nodes = (1 << D) - 1        # 15 internal nodes (depth 4)
n_leaves = 1 << D             # 16 leaves
H = np.zeros(n_nodes)         # thresholds — one per node

# ── 递归设阈值 ──
def set_thresholds(token_ids, node_idx, depth, max_depth=D):
    if depth >= max_depth:
        return  # leaf
    if node_idx >= n_nodes:
        return

    dim = depth % d
    vals = tokens[token_ids, dim]
    median = np.median(vals)
    H[node_idx] = median

    left = token_ids[vals <= median]
    right = token_ids[vals > median]

    if len(left) > 0:
        set_thresholds(left, 2 * node_idx + 1, depth + 1)
    if len(right) > 0:
        set_thresholds(right, 2 * node_idx + 2, depth + 1)

set_thresholds(np.arange(V), 0, 0)

# ── 路由：返回完整路径（每一步的判定）──
def route(token_vec):
    """Returns tuple of 0/1 decisions — this IS the hash."""
    path = []
    i = 0
    for step in range(D):
        dim = step % d
        if token_vec[dim] <= H[i]:
            path.append(0)  # left
            i = 2 * i + 1
        else:
            path.append(1)  # right
            i = 2 * i + 2
    return tuple(path)

# ── 验证 ──
print(f"V={V} d={d} depth={D} nodes={n_nodes} leaves={n_leaves}")
print(f"H: {H.round(3)}")
print()

# 1. Echo = 同 token 同路径
paths = {}
deterministic = 0
for idx in range(V):
    token = tokens[idx]
    p1 = route(token)
    p2 = route(token)
    paths[idx] = p1
    if p1 == p2:
        deterministic += 1

# 2. 路径分布 = hash 碰撞统计
path_counts = Counter(paths.values())
unique_paths = len(path_counts)

# 3. 前缀碰撞（中继节点碰撞）
prefix_shared = 0
for i in range(V):
    for j in range(i+1, V):
        pi, pj = paths[i], paths[j]
        for k in range(1, D):
            if pi[:k] == pj[:k]:
                prefix_shared += 1

print(f"=== Results ===")
print(f"Deterministic  : {deterministic}/{V}")
print(f"Unique paths   : {unique_paths}/{n_leaves} possible")
print(f"Token/path dist: {dict(path_counts)}")
print(f"Prefix collisions (pair×depth): {prefix_shared}")
print(f"Avg colliding pairs per depth prefix: {prefix_shared / (V*(V-1)/2):.1f}")
print()

# 4. 展示碰撞——前 10 个 token 的路径
print("=== Path (hash) samples ===")
for idx in range(10):
    p = paths[idx]
    print(f"  token_{idx:3d}: {p}")
print()

# 5. 同路径 token 组（碰撞组）
path_groups = {}
for idx, p in paths.items():
    path_groups.setdefault(p, []).append(idx)
print("=== Collision groups (same hash) ===")
for p, ids in sorted(path_groups.items(), key=lambda x: -len(x[1])):
    print(f"  {p}: {len(ids)} tokens — {ids[:8]}{'...' if len(ids)>8 else ''}")

assert deterministic == V, f"Not deterministic!"
print("\\n=== ECHO TEST PASSED ===")
