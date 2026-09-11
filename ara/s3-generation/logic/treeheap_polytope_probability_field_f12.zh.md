# F12：TreeHeap 多面概率场几何审计

日期：2026-09-11  
Claim：`S3-TREEHEAP-POLYTOPE-PROBABILITY-FIELD-F12`  
状态：预注册，待 smoke。

## 1. 问题来源

F10 证明有界正交退火在数值上可用，F11 又证明方向参数能够被梯度更新，但同一个标量方向头
在 depth 5、6、7 上表现不一致，且没有形成可迁移的组合协议。一个可能原因是：递归深度只规定
TreeHeap 中的位置，却没有规定该位置只有一个有效方向。数百维 hidden state 在同一深度上可能
同时包含若干相互干涉的语义方向，因此局部概率场更接近有多个面的高维多面体，而不是一根可由
单个角度描述的矢量。

本实验不把“多面体”当作已知事实，而把它降为两个可区分的局部假设：

- 单轴假设：同一 protocol depth、同一 merge depth 的样本梯度近似共线，通道组同步旋转足够；
- 多方向假设：梯度具有多个主方向、明显的负余弦冲突，且分组方向干预优于同范数的同步干预。

## 2. 坐标与干预

沿用 F10 的正交坐标：

\[
c=(l+r)/\sqrt{2},\qquad d=(l-r)/\sqrt{2}
\]

\[
p_g=\frac{c_g+u_gd_g}{\sqrt{1+u_g^2}},\qquad
q_g=\frac{-u_gc_g+d_g}{\sqrt{1+u_g^2}},\qquad
u_g=0.5\tanh(r_g)
\]

base 的 256 个通道分为 8 组，extra 的 384 个通道也分为 8 组。每个 merge depth 因而有
16 个局部方向坐标。模型只递归并暴露 parent `p`；`q` 仅用于验证成对能量闭合，不改变 READ。
所有 `r_g=0` 时必须严格复现当前原生 FOLD。

## 3. 几何测量

固定同一个 E01 TreeHeap-106M checkpoint，冻结全部模型参数。在真实 WMT held-out 样本上，
分别对 protocol depth 5、6、7 计算逐样本 teacher-forced NLL 对每个分组坐标的梯度。由于每个
样本拥有独立坐标，批内 loss 求和后一次反传仍可取得逐样本梯度。

对每个 protocol depth 和 merge depth 记录：

- 样本梯度两两余弦分布及负余弦比例；
- 未中心化第一主方向解释率；
- 梯度矩阵 participation-ratio effective rank；
- 投影到“8 组同步变化”子空间后保留的梯度能量比例；
- base/extra 的梯度范数与逐样本记录。

然后做有限步长复核。对每个样本使用相同的 raw-coordinate L2 位移预算，比较：

1. `scalar`：每个 base/extra、每个 merge depth 的 8 组只能同步移动；
2. `grouped`：全部通道组可沿该样本的完整负梯度分别移动。

这两种移动都由零点的局部梯度决定，再重新前向计算真实 NLL；不以一阶近似代替测量结果。

## 4. 预注册判据

- O0：零坐标 logits 与 NLL 相对 native 的最大差不超过 `1e-7`；
- O1：梯度有限且非零，成对能量相对误差不超过 `1e-5`；
- O2：实验前后冻结 checkpoint 的 trainable state 完全一致；
- P1：至少两个有效的 `(protocol depth, merge depth)` 单元同时满足：第一主方向解释率
  `<0.80`、effective rank `>1.5`、负余弦比例 `>0.10`；
- P2：至少两个 protocol depth 在至少一个预注册步长上，`grouped` 的平均 NLL 比
  `scalar` 低 `1e-4` 以上，且逐样本胜率超过 `0.60`；
- P3：全局同步子空间保留的梯度能量比例低于 `0.80`。

P1--P3 是支持“局部多方向场”的联合证据，不是生产替换门。任一质量指标的非单调都不提前
终止；只有工程合同失败、OOM、CUDA 错误或非有限值才使实验无效。

## 5. 结论边界

若 P1--P3 通过，只能说当前 checkpoint 的局部目标几何不能被单轴角度充分描述，并支持后续
测试分组角、方向专家或 simplex/polytope routing。它不证明概率场字面上是二维多边形，也不
证明这些逐样本、使用 target 得到的 oracle 方向能由推理时模型自行预测。

若梯度近似 rank-1，或同范数 grouped 干预不优于 scalar，则“深度位置上存在多面概率场”在
本次分组尺度上不受支持，应回到训练目标、READ 接口或跨深度共享方式寻找 F11 失效原因。
