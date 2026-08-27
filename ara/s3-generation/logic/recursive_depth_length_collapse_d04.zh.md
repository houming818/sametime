# D04：递归深度与输出长度坍缩探针

日期：2026-08-27

Claim：`S3-RECURSIVE-DEPTH-LENGTH-COLLAPSE-D04`

状态：`partial / exact alignment not supported`；taskd `318` 完成。

## 1. 动机

D03 在相同词表、相同目标长度和相同 Decoder 下逐层开放 READ，证明递归深度会
稳定改变概率场。但它强迫浅层状态预测完整目标句，因此不能回答浅层是否更适合
低分辨率、短输出。

D04 同时改变 READ 深度与输出地址数，观察三组对应关系：

```text
depth 5 -> 1/4 长度
depth 6 -> 1/2 长度
depth 7 -> 1 倍长度
```

## 2. 目标如何缩短

不能把原句直接截成前缀，也不能人工编写摘要。对完整目标 token 序列
`y_0, ..., y_(L-1)`，按连续块构造低分辨率目标：

```text
1/4 长度：每 4 个相邻 token 组成一个局部概率袋
1/2 长度：每 2 个相邻 token 组成一个局部概率袋
1 倍长度：每 1 个 token 组成一个退化概率袋
```

对块 `B_t`，训练目标分布为块内 token 的经验分布：

```text
P_target(v | B_t) = count(v in B_t) / |B_t|
```

本实验冻结参数，没有训练。Decoder 在每个块只输出一个词表概率场；块级 top-1
作为可视化代表 token。其输出长度约为原目标的 `1/4`、`1/2` 或 `1`。

前一个块输入 Decoder 的载体，是该块 token embedding 的均值；第一步仍使用 BOS。
这不是自然语言摘要，而是最小、无人工语义标签的结构坍缩目标。

## 3. 实验矩阵

对每个冻结 checkpoint 运行完整 `3 x 3` 矩阵：

| READ 深度 | block=4 | block=2 | block=1 |
|---|---|---|---|
| 5 | 1/4 输出 | 1/2 输出 | 完整输出 |
| 6 | 1/4 输出 | 1/2 输出 | 完整输出 |
| 7 | 1/4 输出 | 1/2 输出 | 完整输出 |

使用 D03 相同的三个 `read` checkpoint：seed `10101`、`10102`、`10103`。正式实验
固定相同的 256 条 WMT test rows；smoke 使用 seed `10101` 的 64 条。

## 4. 记录量

每个格子记录：

- 按原始 token 数归一化的 soft-target NLL 与 PPL；
- 实际输出地址数及压缩率；
- top-1 是否命中当前 token 块；
- 当前概率场分配给块内任一 token 的概率质量；
- 代表 token 的相邻重复率；
- 固定样例的 1/4、1/2、完整长度读出。

## 5. Predict

### P0：证据合同

- checkpoint 与 test rows 哈希一致，全部统计有限；
- `depth=7, block=1` 与原生完整 READ 的 NLL 差不超过 `1e-6`；
- 三种输出地址数分别接近原目标的 `1/4`、`1/2`、`1`，只允许末块取整误差。

### P1：长度匹配收益

至少两个 seed 同时满足：

```text
NLL(depth=5, block=4) < NLL(depth=5, block=1)
NLL(depth=6, block=2) < NLL(depth=6, block=1)
```

这表示减少输出分辨率能缓解浅层被强迫完成 full-resolution token 任务的问题。

### P2：尺度对应关系

至少两个 seed 的每个深度最佳 block 分别是：

```text
depth 5 -> block 4
depth 6 -> block 2
depth 7 -> block 1
```

P2 是强预测，允许失败。失败表示“较浅层适合较短输出”可能存在，但尚不能声称
当前深度已经自发形成精确的 1/4、1/2、1 尺度协议。

### P3：跨 seed 可复制

将每个 seed 的九格 NLL 排成固定向量。三个 seed pair 中至少两个 Pearson 相关
不低于 `0.90`，才认为长度-深度响应不是单 seed 偶然。

## 6. 结论边界

即使 P1/P2 通过，也只支持结构概率袋的分辨率匹配，不等于人类可读缩句、摘要或
独立层语义。若 P1 通过而 P2 失败，下一步应训练 depth-conditioned collapse
protocol；若 P1 失败，先检查低分辨率目标与 Decoder 历史载体，不直接增加训练量。

## 7. Evidence

```text
ara/s3-generation/evidence/s3_recursive_depth_length_collapse_d04/
```

## 8. 正式结果

P0 与 P3 通过，P1 与 P2 失败。完整输出实现合同成立：`depth=7, block=1`
与原生 READ 的 NLL 差小于 `1e-6`。三个 seed 的九格 NLL 向量 Pearson 相关为
`0.99974`、`0.99980`、`0.99999`，因此结果高度可复制。

三 seed 均值如下：

| READ | 输出尺度 | NLL/token | 块命中率 | 块概率质量 | 重复率 |
|---|---:|---:|---:|---:|---:|
| depth 5 | 1/4 | 31.6560 | 0.0261 | 0.0096 | 0.6446 |
| depth 5 | 1/2 | 31.1372 | 0.0228 | 0.0093 | 0.5843 |
| depth 5 | 1 | 29.4815 | 0.0456 | 0.0332 | 0.5278 |
| depth 6 | 1/4 | 11.5023 | 0.1553 | 0.0411 | 0.3679 |
| depth 6 | 1/2 | 10.8295 | 0.1396 | 0.0433 | 0.3144 |
| depth 6 | 1 | 9.0885 | 0.2180 | 0.1366 | 0.2228 |
| depth 7 | 1/4 | 7.7094 | 0.1814 | 0.0472 | 0.2909 |
| depth 7 | 1/2 | 6.9873 | 0.1657 | 0.0510 | 0.2394 |
| depth 7 | 1 | 5.3003 | 0.2630 | 0.1651 | 0.1507 |

每个深度的最佳 NLL 都来自 `block=1`，而不是预注册的 `4/2/1` 对角线。
短输出地址数正确实现，实际比例约为 `0.269/0.511/1.0`；额外部分来自每句末块
向上取整。但代表 token 大量坍缩为少数高频词，语义没有随长度一起压缩。

固定样例：

```text
source: By using the KMI website, you consent to the data practices described in this statement.
reference: 您在使用 Gerber Technology 网站时，即表明您同意本声明中所述的数据管理和收集方法。

depth 5, 1/4: commercial commercial commercial 但 但
depth 6, 1/2: 但 但 但 但 commercial 但 但 但 但 但
depth 7, full: 我们使用网站 Technology
```

## 9. 解释

该 checkpoint 只训练过“一次 Decoder step 对应一个目标 token”的协议。D04 在
冻结状态下突然要求一步代表 2 或 4 个 token，并把上一块的载体改为 token embedding
均值；这两者均属于模型未见过的输入与目标。因此，D04 否定的是“冻结 checkpoint
可以零训练地使用相邻 token 概率袋完成长度坍缩”，而不是否定低分辨率输出本身。

下一步若继续，应把 block-level collapse 纳入训练，并与逐 token continuation 使用
同初始化、同数据流、同计算预算进行对照。训练前仍需决定：低分辨率目标应是局部
词袋、可学习 latent，还是由目标 TreeHeap FOLD 得到的上层状态。当前证据不支持
直接把相邻词袋作为默认答案。
