"""
SPR Echo Verify — 纯 numpy，加载 H 验证自映射
Usage: python3 spr_echo_verify.py [H_file.npy]
"""
import numpy as np
import sys

D = 4
d = 8
V = 128
n_leaves = 1 << D

def forward_split(scores):
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
    return split(list(range(len(scores))), D)

# 加载 H
if len(sys.argv) > 1:
    H = np.fromfile(sys.argv[1], dtype=np.float32)
else:
    H = np.random.RandomState(42).randn(d).astype(np.float32)
    print(f"using random H")

np.random.seed(42)
tokens = np.random.randn(V, d).astype(np.float32)

scores = tokens @ H
groups = forward_split(scores)
sizes = [len(g) for g in groups]

print(f"H: {H.round(3).tolist()}")
print(f"leaves={n_leaves} sizes={sizes}")
print(f"active={sum(1 for s in sizes if s > 0)}/{n_leaves}")

same = 0
scores2 = tokens @ H
groups2 = forward_split(scores2)
for i in range(V):
    l1 = [j for j, g in enumerate(groups) if i in g][0]
    l2 = [j for j, g in enumerate(groups2) if i in g][0]
    same += (l1 == l2)

print(f"deterministic: {same}/{V}")
print("ECHO OK" if same == V else "FAIL")
