# D07：在递归深度压力下训练 TreeHeap 私有协议

状态：预注册，等待 proof。

Claim：`S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07`

## 1. D06 留下的问题

D06 已经证明，递归深度可以通过有限容量和单调 EOS 压力稳定控制自由生成长度。
但是它只会让句子更早结束，没有告诉模型有限空间里应该保留什么。因此完整深度的 BLEU
从无压力平均 `5.817` 降到 `2.741`。

直接在短输出上继续计算完整目标的逐 token 交叉熵并不合理。超过容量的位置永远无法
出现，梯度最容易学到的是抵消 EOS 压力。把目标截成前缀同样不合理，因为那只会训练
截断，不会训练压缩。

D07 把问题写成一个最小的码率-失真闭环：

```text
source TreeHeap Hx
    -> depth-limited protocol slots Cd
    -> recursive FOLD(Cd)
    -> shared recursive READ/UNFOLD
    -> full target reconstruction
```

递归深度决定码率，完整目标重建误差决定失真。梯度必须在有限容量内决定保留哪些
信息，而不是由人提供缩句标签。

## 2. 压力板

沿用 D06 的源句容量公式。设完整深度 `D=7`，源句除去方向 token 和 EOS 后的有效长度
为 `Lx`：

```text
Bd = max(2, ceil(Lx * 2^(d-D)))
```

本 proof 使用 `d in {5, 6, 7}`，因此容量近似为 `Lx/4, Lx/2, Lx`。为控制 smoke
显存，输入与协议板均限制为 32 个位置；正式实验可以在 smoke 通过后扩展。

`Bd` 只决定可用协议槽位。目标句长度、目标 token 和参考译文都不能参与 `Bd` 或协议
槽位的生成。

## 3. 协议槽位如何形成

每个槽位有一个可学习 query。query 从 source TreeHeap 的 root 开始，逐层执行同一个
递归 READ kernel，并在左右子节点间形成可微分支概率。到指定深度后，最终 query 与
局部 state 合成为一个协议槽位：

```text
q_(k+1) = norm(q_k + K_read(q_k, local_k, depth_k))
c_t     = K_slot(q_final, local_final)
```

同一个 `K_read` 被所有槽位和所有递归深度共享。槽位数量由 `Bd` 限制，超出容量的地址
被 mask 为不可见。

## 4. 槽位仍然是一棵 TreeHeap

协议槽位不是直接交给一个 flat MLP。它们作为 leaf，使用固定、无参数、数值有界的
二叉 FOLD 逐层形成 parent：

```text
parent = (left + right) / sqrt(2)       # 两个子节点都存在
parent = existing_child                 # 只有一个子节点存在
```

由此得到从 root 到 leaf 的完整多分辨率状态。目标 decoder 使用共享的递归 READ kernel
逐层读取这棵协议 TreeHeap，再生成完整目标句。

## 5. 训练隔离

第一阶段冻结 D06 使用的 C12 source TreeHeap checkpoint，只训练：

- 槽位 query 与共享递归压缩 READ；
- 槽位合成 kernel；
- 协议 TreeHeap 的共享递归 READ/UNFOLD decoder。

冻结 source encoder 的目的不是最终架构设计，而是证明有限容量协议本身能否学习。
若一开始联合训练全部参数，source encoder 可能重写 H 来绕过瓶颈，我们将无法判断收益
来自压力协议还是来自普通微调。

唯一主损失为完整目标的 token 交叉熵：

```text
L_distortion = CE(decoder(FOLD(Cd)), target)
```

每个 batch 在深度 `5/6/7` 中均匀采样。没有摘要标签、目标前缀标签或人工逻辑标签。

## 6. Predict 与证伪条件

### P0：实现合同

- source checkpoint 冻结且训练前后哈希不变；
- 压缩器不读取 target；
- 目标长度不参与容量计算；
- 参数中不存在 Transformer、自注意力或 flat `L x L` 路由表；
- 三个深度共享压缩 READ、FOLD 和重建 decoder 参数。

### P1：可训练性

smoke 结束后，至少两个深度的 valid NLL 比各自初始化下降 `>= 0.10`，所有梯度与状态
均为有限值。

### P2：压缩态具有输入因果性

在同一冻结 test 集上，跨样本打乱协议槽位或把槽位置零，都应使平均 NLL 比 native
至少增加 `0.10`。否则 decoder 主要依赖自己的语言偏见，不能说协议承载了 source
信息。

### P3：码率-失真方向

训练后平均 NLL 应满足：

```text
NLL(depth=7) <= NLL(depth=6) <= NLL(depth=5) + 0.05
```

这不是要求每个样本严格单调，而是要求容量增加没有系统性伤害重建。

### P4：协议没有退化成常量

native 槽位的 batch 方差大于 `1e-4`，跨样本 cosine 不得全部接近 1；压缩 READ 与
重建 READ 的梯度范数都必须大于 0。

### P5：生成边界

记录每个深度的自由生成、BLEU、重复率与固定样例，但 smoke 不以 BLEU 达标作为
Claim 成立条件。D07 首先验证“有限容量私有协议能否被梯度写入并被递归读出”，不把
它提前宣称为人类可读缩句。

## 7. 决策

- P0 失败：实现无效，停止。
- P1 或 P2 失败：压力板尚未形成可用私有协议，停止正式扩容。
- P1/P2 通过而 P3 失败：协议可学，但当前容量/FOLD 规律错误，只允许修改结构后重试。
- P0-P4 通过：允许进入多 seed、更多 steps 的正式训练，再评估生成质量。

Evidence 目录：

```text
ara/s3-generation/evidence/s3_recursive_depth_pressure_protocol_d07/
```

## 8. D07 smoke 结果

taskd `328` 在 seed `10701` 上完成 120 steps。实现合同、可训练性、码率-失真方向与
非退化检查通过，但最关键的输入因果 Gate P2 失败，因此没有进入正式训练。

三个深度的 valid NLL 都显著下降：

| depth | 初始化 valid NLL | 训练后 valid NLL | 下降 |
|---:|---:|---:|---:|
| 5 | 21.865 | 8.291 | 13.574 |
| 6 | 23.473 | 8.006 | 15.467 |
| 7 | 25.018 | 7.987 | 17.030 |

但是 test 因果消融给出了反方向结果：

| depth | shuffle 相对 native NLL | zero 相对 native NLL |
|---:|---:|---:|
| 5 | +0.060 | -1.902 |
| 6 | +0.056 | -1.523 |
| 7 | +0.067 | -1.513 |

跨样本打乱只有很弱的损失，而把协议槽位全部置零反而明显降低 NLL。自由生成也主要
由重复地名与未知词片组成，BLEU4 约为 `0.27-0.29`。因此 NLL 的大幅下降不是私有
协议成功，而是放开的重建 decoder 学会了目标语料的边际语言偏见；当前协议 state
反而成为干扰。

最终结论：

```text
smoke_blocks_formal
```

这个负结果保留。D07R1 将冻结继承的语言骨架，只训练压力协议两端的递归映射，以
排除 decoder 单独学习目标词频的捷径。
