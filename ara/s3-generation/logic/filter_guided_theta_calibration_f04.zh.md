# F04：滤镜引导的退火参数校准

日期：2026-09-08
Claim：`S3-FILTER-GUIDED-THETA-CALIBRATION-F04`
状态：预注册，尚未取得结果。

## 1. 问题

F03 的逐节点滤镜证明：冻结 TreeHeap 的运行态隐节点幅度会因果改变直接 Decode，而且不同
节点的敏感度不同。后续的重复词探针又表明，`push / like / want` 的词汇方向可以被放大到
可读状态，但加入对象或人物后经常消失或发生角色错绑。

F04 不把滤镜当成最终架构参数。滤镜只作为干预探针，寻找当前隐态附近更低行为损失的状态；
真正训练的是计算 parent 隐态的退火参数 `theta`。实验要回答：滤镜找到的局部修正能否被
共享退火方程吸收，使撤掉滤镜后的原生隐态保留更多组合词汇信号。

## 2. 参数化退火方程

当前固定退火为：

```text
p0 = (left + right) * sqrt(0.5)
```

F04 为 base 256 维通道和 extra 384 维通道分别增加一个低秩、内容条件化的径向增益：

```text
feature = tanh(P * [LN(left), LN(right), LN(left - right)])
raw_d = bias_d + <head_d, feature> / sqrt(rank)
gain_d = exp(log(1.5) * tanh(raw_d))
parent = p0 * gain_d
```

`head_d` 与 `bias_d` 初始化为严格的 0，因此 step 0 的 `gain_d` 严格为 1，输出应与原
checkpoint 一致。增益被限制在 `(2/3, 1.5)`。参数在相同深度的所有节点间共享，但可根据
左右子状态生成不同增益；它不是 63 个永久地址参数。

## 3. 行为损失

不使用完整参考译文。对 Decoder 每个时间位置的 softmax，计算目标词形在整段固定时域中
至少出现一次的可微覆盖概率。正词要求出现，负词要求不出现，同时加入 EOS 覆盖与相邻重复
概率惩罚。

受控概念为 `push / stone / like / apple / want`。训练输入包含裸词重复与动宾重复；测试输入
加入未在训练集合出现的“西西弗”和“小明”组合。这个目标只能评价词汇守恒，不能证明语序、
角色绑定或完整翻译已经成立。

## 4. 滤镜引导

每个 batch 的内层从全 1 临时滤镜开始，只允许改变 internal parent，不改变 leaves。内层
使用同一行为损失寻找附近的节点幅度修正 `W*`，并在结束后丢弃。

外层有两个同初始化、同训练步数的臂：

| 臂 | 更新方式 |
|---|---|
| `direct-theta` | 只用原生无滤镜行为损失更新退火 `theta` |
| `filter-guided-theta` | 原生行为损失，加上原生 logits 向临时 `W*` logits 的 KL 蒸馏 |

两臂都冻结原 checkpoint 的全部 105,965,594 个参数，只训练新增退火方程。这样不会把收益
混同为 Decoder 或词嵌入重新记忆样本。

## 5. Smoke 合同

- checkpoint：E01 TreeHeap-106M pass 2，step 132873；
- depth `5/6/7` 循环；
- 固定行为时域 24 tokens，直接 Decode 最长 64 tokens；
- 两臂各 30 outer steps；guided 每步 3 个 inner filter steps；
- AdamW，只更新退火 `theta`；
- 允许质量非单调，只有 OOM、CUDA/Xid、NaN/Inf、冻结参数改变、step-0 不同起点或证据损坏
  可以判为无效。

Smoke 只验证实现、梯度与可观测方向，不据 30 步结果替换默认退火。

## 6. 预注册判定

### P0：严格同起点

退火 `theta=0` 时，受控样本直接 Decode 与原 checkpoint 逐 token 一致，固定 WMT valid
mean NLL 差不超过 `1e-9`。

### P1：滤镜确实提供局部方向

step 0 内层滤镜后的训练行为损失低于全 1 滤镜；否则“滤镜可以充当局部老师”未得到支持，
但仍保存 direct 臂结果。

### P2：theta 可训练且基座冻结

两臂退火参数均收到非零有限梯度，所有新增参数有限，原 checkpoint 参数逐 tensor 不变，
保存后重载的评价差异不超过浮点复现范围。

### P3：撤掉滤镜后的训练集词汇守恒

相对共同 step 0，原生无滤镜 train positive coverage 提高，且 negative activation 与 expected
repetition 均不恶化超过 `0.05`。

### P4：未见组合迁移

test positive coverage 相对 step 0 提高；只有 `filter-guided-theta` 比 `direct-theta` 高至少
`0.02`，才支持“滤镜引导提供了普通行为训练之外的增量”。不满足时不能把 direct 改善归功于
滤镜。

### P5：滤镜遗憾值

对训练后原生隐态重新寻找临时滤镜。若 `native_loss - filtered_loss` 相对 step 0 下降，说明
退火参数吸收了一部分原本需要外挂滤镜才能得到的修正；否则记录为未吸收。

## 7. 边界

标量滤镜只测得梯度在现有隐态方向上的径向投影，不能完整告诉 `theta` 应向哪个向量方向
旋转。F04 的退火方程因此也只学习内容条件化径向校准。若该阶梯有效，后续才考虑通道级或
低秩矩阵方向修正；若无效，不能据此否定所有可训练退火，只能否定当前径向参数化和行为目标。
