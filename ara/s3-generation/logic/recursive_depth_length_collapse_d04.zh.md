# D04：递归深度与输出长度坍缩探针

日期：2026-08-27

Claim：`S3-RECURSIVE-DEPTH-LENGTH-COLLAPSE-D04`

状态：预注册。

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

