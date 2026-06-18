# S1: Echo — Semantic Prefix Routing

## Core Question

能否用递归语义判定树（SPR TreeHeap）替代 Transformer 输出层的稠密矩阵，实现高效多义消歧和跨语言对齐？

## Hypothesis

词汇语义不是单点嵌入，而是叠加态。通过堆树（Heap Tree）上的路径路由，可以在保持高熵多义的同时，利用上下文（L1）做确定性坍缩。

## Design

```
L0 (词法基底): token → 堆树路径 → 128D 多义叠加向量
L1 (上下文消歧): CMul(语义向量, 树路径) → 坍缩到确定语义流形
```

## Key Constraints

- 不使用 Transformer attention
- 32K 词汇表 → 31 共享节点树（5层二叉）
- CMul（复平面乘法）组合路径节点
- InfoNCE 损失做锚点对齐

## References

- S1 blogs: `blogs/*/spr/s1-echo-session.md`, `001*.md` ~ `007*.md`
- Code: `/data/homecicd/sametime/code/wmt/core.py` (TreeNodes), `spr_anchor_bridge.py`
- Checkpoint: `/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep*.pt`
- Provides: 128D L0/L1 vectors → consumed by S2
