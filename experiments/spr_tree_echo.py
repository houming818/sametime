"""
SPR Tree Echo — 固定二分 + 每层共享参数
echo test: 全1参数 → output=input → loss=0
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

tokens = np.array([1.0, 2.0, 1.0, 2.0])
depth = 2
params = [np.ones(4), np.ones(2)]

output = tree_forward(tokens, params, depth)
loss = ((output - tokens)**2).mean()

print(f"input : {tokens}")
print(f"output: {output}")
print(f"loss  : {loss}")
print(f"echo  : {np.allclose(tokens, output)}")

# 分裂轨迹
x = tokens.copy()
for lvl in range(depth):
    p = params[lvl][:len(x)]
    x = x * p
    print(f"  L{lvl}: *{p} → {x}")
    x = x[:len(x)//2]
