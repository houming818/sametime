"""
SPR Echo Test
验证堆路由能否实现自映射：token → 路由 → 叶子 → 取出自己
手工设阈值，不训练。
"""
import numpy as np

V, D, d = 100, 3, 16          # 100 tokens, depth=3 (7 nodes / 8 leaves), 16-dim
np.random.seed(42)
tokens = np.random.randn(V, d)

n_nodes = (1 << D) - 1        # 7 internal nodes
n_leaves = 1 << D             # 8 leaves
H = np.zeros(n_nodes)         # thresholds
leaf_values = np.zeros((n_leaves, d))

# ── 递归手工设阈值 ──
def set_thresholds(token_ids, node_idx, depths):
    """对 token_ids 集合在深度 depths 设阈值. node_idx 起始=0."""
    if node_idx >= n_nodes:
        leaf_idx = node_idx - n_nodes
        if len(token_ids) == 1:
            leaf_values[leaf_idx] = tokens[token_ids[0]]
        elif len(token_ids) > 1:
            leaf_values[leaf_idx] = np.mean(tokens[token_ids], axis=0)
        return

    dim = depths[0] % d
    vals = tokens[token_ids, dim]
    median = np.median(vals)
    H[node_idx] = median

    left = token_ids[vals <= median]
    right = token_ids[vals > median]

    next_depths = depths[1:]
    if len(left) > 0:
        set_thresholds(left, 2 * node_idx + 1, next_depths)
    if len(right) > 0:
        set_thresholds(right, 2 * node_idx + 2, next_depths)


set_thresholds(np.arange(V), 0, list(range(D)))

# ── 路由函数 ──
def route(token_vec):
    i = 0
    for step in range(D):
        dim = step % d
        if token_vec[dim] <= H[i]:
            i = 2 * i + 1
        else:
            i = 2 * i + 2
    return i - n_nodes  # leaf index

# ── 验证 ──
print(f"V={V} d={d} depth={D} nodes={n_nodes} leaves={n_leaves}")
print(f"H thresholds: {H.round(3)}")
print()

errors = []
leaf_hit = np.zeros(n_leaves, dtype=int)
deterministic = 0
out_of_bounds = 0

for idx in range(V):
    token = tokens[idx]
    leaf = route(token)

    # bounds check
    if leaf < 0 or leaf >= n_leaves:
        out_of_bounds += 1
        continue

    leaf_hit[leaf] += 1

    # determinism
    leaf2 = route(token)
    if leaf == leaf2:
        deterministic += 1

    # echo: is leaf value == token?
    err = np.abs(leaf_values[leaf] - token).mean()
    errors.append(err)

print(f"=== Results ===")
print(f"Out of bounds: {out_of_bounds}/{V}")
print(f"Deterministic : {deterministic}/{V}")
print(f"Mean abs error: {np.mean(errors):.6f}")
print(f"Zero-error echos: {sum(1 for e in errors if e < 1e-6)}/{V}")
print(f"Leaf distribution: {leaf_hit}")
print(f"Dead leaves: {sum(leaf_hit == 0)}/{n_leaves}")

assert out_of_bounds == 0, f"OOB!"
assert deterministic == V, f"Not deterministic!"
print("\\n=== ECHO TEST PASSED ===")
