# 真实 WMT 退火式 Token 空间实验报告

## 结论

真实语料桥接实验支持以下有限结论：

> TreeHeap 不需要随机初始化的 token embedding 表，也能从真实 WMT
> token-context 概率律中形成非随机、可复现、具有预测能力的路径坐标。

但本实验**不支持 TreeHeap 已优于平坦聚类或 SGNS**。TreeHeap 与 flat
K-means 的 held-out NLL 基本打平；SGNS 的预测和可读邻域仍然更强；并且
TreeHeap 在一个 bootstrap seed 上出现了分支占用坍缩。

## 数据合同

```text
语料：WMT massive 中英平行语料的英文侧
扫描行数：200,000
有效行数：200,000
SentencePiece token：4,283,059
目标 token：512 个高频、可读英文 word-start piece
context token：1,024
窗口：左右各 4
训练 context pair：6,599,588
测试 context pair：735,369
bootstrap seeds：19101..19108
TreeHeap 深度：5，对应最多 32 个 leaf
```

测试行由 `line_index mod 10` 预先固定，没有参与路由形成。

## 对照组

1. `frequency_shuffle`：只在相邻频率桶内交换 token 的 context 行。
2. `random_route`：保留真实观测，但随机生成深度 5 路径。
3. `flat_kmeans`：相同 Hellinger 输入、相同 32 个 cluster，但没有树约束。
4. `expected_sgns`：32 维期望负采样基线。
5. token 经验分布和 global unigram 作为信息上下界参考。

## 主要结果

| 模型 / 对照 | Held-out context NLL | 解释 |
|---|---:|---|
| TreeHeap 5D 路径 | **5.35895** | 当前方案 |
| 频率匹配打乱 | 5.82061 | 破坏 token-context 身份 |
| 随机路径 | 5.46101 | 保留数据，破坏路由 |
| flat K-means | 5.36257 | 与 TreeHeap 基本打平 |
| expected SGNS 32D | **5.20325** | 当前最强学习基线 |
| token 经验参考 | 5.20890 | 不压缩的 token 行 |
| global unigram | 5.56978 | 不区分 token |

TreeHeap 从 global `5.56978` 降到 `5.35895`，获得了 global 到 token 经验
参考全部可用改善量的大约 `58.4%`。随机路径只能获得约 `30.1%`。

## 路径是否只是随机编码

| 指标 | TreeHeap | 频率打乱 | 随机路径 | flat K-means |
|---|---:|---:|---:|---:|
| 跨 bootstrap co-cluster 稳定性 | **0.35096** | 0.08583 | 0.03122 | **0.59186** |
| 与 SGNS top-10 邻域重合 | **0.17148** | - | 0.02014 | 0.12593 |
| 频率解释度 R2 | 0.31957 | 0.35551 | 0.06558 | 0.25764 |

TreeHeap 与 SGNS 的邻域重合约为随机路径的 `8.5` 倍，因此路径不只是
随机编号。频率 R2 为 `0.32`，说明坐标确实包含频率成分；但频率打乱臂
R2 更高、NLL 却明显更差，所以频率不能单独解释预测收益。

## 可读样例

TreeHeap 的邻域已经出现真实结构：

```text
of    -> in, within, between, by, from, under, at, on
people -> now, who, way, because, when, they, They, would
time  -> many, being, every, work, his, both, Chinese, full
```

但它仍明显弱于 SGNS。例如：

```text
SGNS(people) -> children, things, those, countries, they, them, country, world
SGNS(time)   -> day, first, end, form, real, year, after, life
```

`water` 的 TreeHeap 邻域仍混入 `power, project, market` 等宽泛词，说明五层
路径已经捕获共现结构，但还不是成熟的词义空间。

## 暴露的问题

### 1. 树形优势没有成立

TreeHeap NLL `5.35895` 与 flat K-means `5.36257` 的差仅 `0.00362` nat，
小于 bootstrap 波动，不能宣称 TreeHeap 更优。flat 的跨 seed 稳定性
`0.59186` 还明显高于 TreeHeap 的 `0.35096`。

### 2. 一个 seed 发生部分坍缩

seed `19105`：

```text
leaf utilization = 0.71875
occupancy entropy = 0.50932
TreeHeap NLL = 5.40765
```

其他七个 seed 的利用率为 `0.875..1.0`。这说明当前递归二分在某些
bootstrap 扰动下会进入不均衡分叉。均值门通过，但算法鲁棒性尚未完成。

### 3. SGNS 仍保留更多信息

SGNS 使用 32 维连续坐标，TreeHeap 当前只有 5 个路径概率。TreeHeap 的
压缩更强，但相对 SGNS 损失约 `0.156` nat。下一步不能简单宣布替代
embedding，而应研究增加路径容量、detail/residual 或 context-conditioned
route 后能否缩小这个差距。

## 判定

```text
真实语料中存在可退火的 token-context 结构：支持
TreeHeap 路径不是随机/纯频率结果：支持
TreeHeap 优于 flat clustering：不支持
TreeHeap 达到 SGNS 表示质量：不支持
当前分裂过程对 bootstrap 稳定：部分不支持
```

下一次实验应直接修复分支质量约束，重点观察 seed `19105` 类型的坍缩，
然后再测试 `route(token, context)` 处理一词多义。单纯增加语料不会自动
解决这个拓扑不稳定问题。

## 完整性

```text
scan SHA-256: cee68aef5b76cc40a9ee0fd7f34c2dff9e1ac0dd19be49cd5ba43118dcb4fd4d
SPM SHA-256: 9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731
records: 8/8
non-finite values: 0
io task: 449
task elapsed: 92.7 seconds
```
