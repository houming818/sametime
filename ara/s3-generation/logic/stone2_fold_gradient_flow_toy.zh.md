# STONE-2 退火 FOLD 梯度流 toy

日期：2026-08-23

类型：公式审计，不是语言训练 Claim。

## 问题

当前零点参考系 FOLD 保证 parent 有界并可由 `root + details + scales`
精确恢复，但前向有界不自动保证反向梯度等能级。本 toy 在完全可控的向量上回答：

1. 共同缩放 leaf 时，parent 是否不变、Jacobian 是否按反比例变化；
2. 同向、正交、交替抵消和随机 leaf 递归后，各深度梯度如何到达 leaf；
3. 当前共享 READ kernel 从多个深度接收的直接梯度是否天然同向；
4. `read_gain = sigmoid(-4)` 下，多层残差修正如何随深度积累。

## 固定公式

对左右向量 `a,b`：

```text
s = sqrt(||a||^2 + ||b||^2 + 1e-8)
parent = (a + b) / (sqrt(2) * s)
detail = (b - a) / (sqrt(2) * s)
```

READ toy 使用与当前实现相同的：

```text
Linear(3d, 2d) -> GELU -> Linear(2d, d) -> tanh
q[d+1] = q[d] + sigmoid(-4) * K(q[d], local[d], depth[d])
```

## 预期

- parent norm 不超过 1，FOLD/UNFOLD 闭包误差接近浮点极限；
- `a,b` 同时乘正数 `c` 时 parent 基本不变，而 parent Jacobian 约按 `1/c` 缩放；
- root 到 leaf 的梯度由沿路径 Jacobian 连乘决定，因此对输入尺度和局部退化敏感；
- leaf 直接通道不经过该连乘，天然比单个 parent 修正更强；
- 共享 READ 参数的深度梯度余弦不受公式保证为正，Gram 矩阵可直接显示冲突位置。

## 边界

随机 READ 权重只用于暴露计算图性质。任何梯度夹角都不能解释为语言语义，也不能
直接升级 C03/C04。若 toy 显示条件数或梯度比例存在系统性问题，后续才为该数学
问题注册 matched smoke。

## 结果

io 任务 `295` 使用 float64、8 leaves、4 dimensions 完成。

### 确定性性质

- 所有 parent norm 均不超过 1；
- 最大 FOLD/UNFOLD 闭包误差为 `4.26e-14`；
- aligned leaf 从尺度 `0.01 -> 1 -> 100` 时，parent norm 基本保持 1；
- 对应 parent Jacobian operator norm 为 `70.7089 -> 0.7071 -> 0.007071`，
  严格呈约 `1/c` 缩放。

因此，“前向尺度不变、反向梯度尺度不变”并不同时成立。共同缩放虽然不改变
parent，却会反比例改变从 parent 回到 leaf 的梯度强度。

### 递归路径

同一个 root 标量探针回传到 leaf 的梯度范数：

| leaf 模式 | root-to-leaf 梯度范数 |
|---|---:|
| coherent，尺度 1 | 0.3476 |
| coherent，尺度 0.01 | 34.7611 |
| coherent，尺度 100 | 0.003476 |
| orthogonal cycle | 0.2582 |
| random unit | 0.6850 |
| alternating `a,-a` | 70,710,677.94 |

alternating 模式第一层 parent 全部为零；后两层 scale 都落到
`sqrt(1e-8)=1e-4`。上层 Jacobian 中的 `1/s` 连乘造成约八个数量级的梯度
放大。该结果说明：当前公式的 forward boundedness 和 exact closure 不能推出
backward conditioning。反对称信息虽然安全地存在 detail 中，但“只把 parent
方向继续向上递归”会在共同方向为零时形成近奇异点。

### READ

四个深度全部读取后，query correction norm 约为 `0.023` 到 `0.049`。在这个
固定随机初始化和统一能量目标下，各深度 READ 参数梯度余弦全部为正：最小值
约 `0.078`，负余弦比例为 0。

所以 toy 不支持“共享 READ 必然造成深度梯度冲突”。C03 checkpoint 中观察到的
负交互可能来自训练后的 state、depth embedding、递归 query 或 routing，必须对
真实 checkpoint 直接计算逐深度梯度 Gram，不能归罪于共享 kernel 形式本身。

## 当前数学判断

下一步优先级应从“修改 S7 统计口径”前移到两个更直接的问题：

1. 测量真实 checkpoint 中各层 scale 与 FOLD Jacobian 条件数，确认近零 parent
   是否在自然数据中实际出现；
2. 在同一 batch 上分离各深度对 READ、Butterfly 和 embedding 参数的梯度，计算
   Gram 矩阵，定位负交互究竟产生在哪个参数组。

只有自然数据也出现近奇异梯度时，才注册带尺度下限或 scale-carrier 的新 FOLD。
toy 本身不足以决定修改公式。
