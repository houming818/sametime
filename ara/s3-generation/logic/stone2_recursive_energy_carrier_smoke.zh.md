# STONE-2 递归能量载体 smoke

状态：预注册后执行

## 问题

当前零点参考系 FOLD 在每一层临时计算

```text
s = sqrt(||left||^2 + ||right||^2 + epsilon)
parent = (left + right) / (sqrt(2) * s)
```

`s` 会保存给 UNFOLD，但下一层 FOLD 只接收 `parent`。因此，当两个高能量
子树在共同方向上抵消时，下一层无法区分“没有输入”和“强信号抵消”，其局部
尺度可能直接落到 `sqrt(epsilon)`。

本 smoke 检查：让节点额外携带绝对能量，能否消除抵消造成的递归近奇异点；
以及直接把左右能量相乘是否反而破坏有界性。

## 三条对照臂

### A. local-current

完全复现当前公式。每层从当前两个方向状态重新计算局部尺度，旧尺度不进入
下一层。

### B. energy-carrier

每个节点携带 `(u, E)`：`u` 是方向状态，`E` 是该子树的绝对能量。leaf 初始化：

```text
E_leaf = sqrt(||x||^2 + epsilon)
u_leaf = x / E_leaf
```

合并时：

```text
E_parent = sqrt(E_left^2 + E_right^2)
parent = (E_left * u_left + E_right * u_right)
         / (sqrt(2) * E_parent)
detail = (E_right * u_right - E_left * u_left)
         / (sqrt(2) * E_parent)
```

绝对能量也可以写成路径乘积：

```text
E_child = E_parent * r_child
r_child = E_child / E_parent
```

因此从 root 到 leaf 的尺度由各层无量纲比例连乘得到；实现同时检查这个乘法
闭包。这里的乘法用于纵向尺度传播，不取代同层平方和。

### C. geometric-product

作为直接乘法反例，局部尺度使用：

```text
s = sqrt(||left|| * ||right|| + epsilon)
```

该臂仍保存 `s`，所以可以代数 UNFOLD；但预期在左右能量严重失衡或一侧为零时
失去 parent 有界性。

## 数据

使用 8-leaf、4-dimensional、float64 的确定性树：

- coherent：所有 leaf 同向；
- alternating：`a,-a` 交替，强信号精确抵消；
- imbalanced：左右幅度在 `100` 与 `0.01` 之间交替；
- one-sided-zero：每对中一个 leaf 为零；
- random-unit：固定 seed 的单位随机向量；
- small/large coherent：整体尺度分别为 `0.01` 与 `100`。

## 测量

1. 所有层 parent 最大 norm；
2. FOLD/UNFOLD 最大绝对闭包误差；
3. root 标量探针对 leaf 的梯度 norm 和最大绝对值；
4. 各层局部尺度或绝对能量范围；
5. energy-carrier 的 root-to-leaf 比例连乘重建误差；
6. 共同缩放 `0.01/1/100` 时的 parent 与梯度变化。

## 预测

- P1：A 在 alternating 上重现约 `1/epsilon` 量级的递归梯度放大；
- P2：B 在 alternating 上把梯度恢复到与 coherent 同一数量级，同时保持闭包；
- P3：B 仍保留归一化方向固有的整体 `1/c` 梯度缩放；它只修复抵消造成的
  额外递归放大，不宣称解决全部条件数问题；
- P4：C 在 imbalanced 或 one-sided-zero 上出现远大于 1 的 parent norm；
- P5：B 的路径比例连乘能重建 leaf 能量，误差低于 `1e-10`。

## 判定边界

这是数值微积分 smoke，不是语言、语义或产品质量证据。即使 B 通过，也只能
支持把它注册为正式 checkpoint 审计与训练 ablation 的候选，不能直接替换现有
FOLD。若 B 不能同时保持闭包、抵消稳定性和乘法尺度闭包，则停止该公式路线。

## Smoke 结果

执行环境：io taskd，8 leaves x 4 dimensions，float64，seed `16211`。

| 输入 | 公式 | 最大 parent norm | root-to-leaf 梯度 norm |
|---|---|---:|---:|
| coherent | local-current | 1.0000 | 0.34761 |
| coherent | energy-carrier | 1.0000 | 0.34761 |
| alternating | local-current | 0 | 70,710,677.94 |
| alternating | energy-carrier | 0 | 0.35355 |
| imbalanced | geometric-product | 70.71775 | 0.00983 |
| one-sided-zero | geometric-product | 7,071.0678 | 0.98319 |

energy-carrier 在所有模式上的 FOLD/UNFOLD 最大误差不超过约 `2.84e-14`，
root-to-leaf 能量比例连乘重建误差为 0。它消除了“强信号抵消后，上层只剩
epsilon”导致的额外递归梯度放大。直接用左右 norm 的几何乘积做局部分母则未
保持 parent 有界，P4 的反例成立。

但该候选没有解决归一化方向本身的整体尺度条件数：coherent 输入整体尺度
`0.01/1/100` 时，其梯度 norm 仍约为 `34.759/0.3476/0.003476`，继续呈
`1/c` 变化。因此 smoke 支持的是一个窄结论：

> 纵向携带绝对能量并沿地址连乘相对比例，可以修复抵消造成的递归近奇异点；
> 它不能单独修复所有归一化梯度问题，也尚未证明有利于语言训练。

下一步若继续，应在固定 checkpoint batch 上先做旁路审计：测量自然数据中
抵消事件频率、候选 Jacobian 和逐深度梯度 Gram。只有确认现有公式的近奇异点
真实进入训练，再注册同初始化、同数据、同 steps 的 FOLD ablation。
