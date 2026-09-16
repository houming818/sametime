# 退火式 FOLD/UNFOLD Token 空间实验报告

## 结论

`S1-ANNEAL-SPACE-C01` 在受控语料上得到支持。

这次证明的不是“TreeHeap 已经理解自然语言”，而是一个更小、但关键的
存在性结论：**不使用随机初始化、可训练的 token embedding 表，仅依据
token-context 概率观测，TreeHeap 的 FOLD/UNFOLD 退火过程可以生成稳定、
非随机、具有类别结构的 token 路径坐标。**

正式任务为 io `task 446`，运行 24 个 seed，共得到 72 条完整 arm 记录。
九个预注册判据全部通过。

## 算法

每个 token 首先被表示为观测到的条件概率：

```text
x_i[c] = P(context=c | token=i)
```

算法没有创建 `nn.Embedding(token)`。它对 `sqrt(x_i)` 做递归二分：

```text
UNFOLD: 根据 token 到左右 child 原型的距离，计算软路由概率
FOLD:   按软路由责任重新求左右 child 原型
COOL:   逐步降低温度，使软路由变得明确
```

每一层“走向右 child 的概率”组成 token 的路径坐标。分裂初始方向由当前
节点观测矩阵的第一奇异方向确定，不依赖随机摆放 token。

## 三个实验臂

| 实验臂 | 输入概率律 | 路由 |
|---|---|---|
| structured | 保留类别共现规律 | 退火 FOLD/UNFOLD |
| shuffled | 每个 token 的样本量不变，但 context ID 被独立打乱 | 同一退火算法 |
| random_route | 保留真实结构 | 随机路径 |

类别标签只用于实验结束后的审计，没有进入训练或分裂过程。

## 正式结果

| 指标 | structured | shuffled | random_route |
|---|---:|---:|---:|
| held-out context NLL | **2.7745** | 8.6505 | 3.6886 |
| cluster purity | **1.0000** | 0.3991 | 0.3906 |
| pairwise F1 | **0.7438** | 0.0740 | 0.0850 |
| route kNN top-3 | **0.8902** | 0.1148 | 0.1105 |
| leaf utilization | 0.9948 | 0.9974 | 0.9714 |
| occupancy entropy | 0.9441 | 0.9760 | 0.9471 |
| route stability | **0.7094** | 0.2296 | 0.2368 |

补充参考：token 自身的经验分布 NLL 为 `2.6289`，全局 unigram NLL 为
`4.5731`。TreeHeap 的 `2.7745` 接近 token 级上限参考，同时明显优于全局
词频和随机路由。

## 如何解释

1. **不是“叶子占满了所以看起来好”。** 三个臂的叶利用率都接近 1，
   但只有 structured 臂得到高 purity、pairwise-F1 和 held-out 预测能力。
2. **概率律提供了 token 身份。** 打乱 context 后，完全相同的退火算法
   失去类别结构，说明结果不是树的固定拓扑自动制造出来的。
3. **FOLD 数值守恒。** structured 臂 parent 与按质量加权 children 的
   最大绝对误差均值为 `1.02e-16`，说明递归没有靠能量爆炸形成分离。
4. **纯度 1.0 不代表层次已经完美。** pairwise-F1 只有 `0.7438`，说明
   每个 leaf 内部没有混入别类 token，但同一类别可能被拆到多个 leaf。
   当前结果是“正确分离但存在过分割”。

## 证据边界

本实验使用的是明确可审计的受控概率律，因此只能支持算法存在性。它还
不能说明：

- WMT 或自然文本会自动形成同样稳定的路径；
- TreeHeap 优于 SGNS/NCE 或平坦聚类；
- 这些路径可以被 decoder 直接读成句子；
- 一词多义能够由单一路径坐标处理。

## 下一步

下一档不应立刻替换现有 embedding，而应做真实语料桥接实验：从实际
共现矩阵产生路径，并加入频率匹配、flat clustering、SGNS/NCE 三类基线。
同时观察多义词是否需要 `route(token, context)`，以及 coarse 节点是否能
提供独立于 fine leaf 的可读信息。

## 完整性

```text
seeds: 19001..19024
records: 72 = 24 seeds x 3 arms
non-finite values: 0
summary SHA-256: ae7535563cd51748d3dc70aecbf156dd2d47acb8d5c3feec71f74de3c933b82f
results SHA-256: bcf302742f22acd4fddd3d72c290ac1895ff4b94b4255b10b08de00bc2dface0
trace SHA-256: a147e1e4cd5e46c1d99a274b7b4e75c46dae56b2dfb5c2e38c0434d6ba623262
taskd log SHA-256: 1ee717a81682ba0a4a5251a38833036ce2ce834f5b64f09e5588793dad09a965
source SHA-256: 1d9d4d6a5af45ca3b233c864fdb295779d7ab981c6d69f217bea96ef38227503
```
