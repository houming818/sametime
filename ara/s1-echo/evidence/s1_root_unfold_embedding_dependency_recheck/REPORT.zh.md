# Root-Unfold Embedding 旧 Claim 依赖复核

## 问题

新的 root-unfold embedding 已经能从真实语料的 token-context 条件分布中
形成非随机坐标。此前失败或弱阳性的 Claim 中，哪些确实受制于随机
embedding，替换以后能否改善？

本次只重测两个直接依赖：

1. `S1-COMPACT-CONTENT-ROUTE-C01`：旧实验使用固定随机 token 向量，
   再把子树 token 向量相加。
2. `S1-TH-EMBED-FP-C01`：旧实验从随机可训练 embedding 开始，让
   embedding 与 TreeHeap 在 pair-NCE 下共同更新。

MiniLM 外部标尺、SGNS 对照以及完整翻译模型没有直接替换，因为那会改变
原 Claim 或整个 encoder/decoder 协议。

## 结果一：Compact Route 变差

在完全相同的 20K WMT、词表 1024、64 维、seed 42 和五轮训练下：

| 输入坐标 | OOD step acc. | OOD route exact |
|---|---:|---:|
| 固定随机向量 | `0.99710` | `0.98260` |
| root-unfold embedding | `0.89200` | `0.48199` |

这不是新 embedding 自己坍缩。进入 compact route 前，它的质量审计为：

- 质量守恒误差：`1.19e-7`；
- 64 个维度中使用 54 个，利用率 `0.84375`；
- 归一化占用熵：`0.87370`；
- 可训练 token 专属参数：`0`。

真正的问题是组合运算不相容。随机归一向量彼此近似正交，子树求和后，
内积很容易判断某 token 是否存在。root-unfold 向量是共享概率坐标，直接
相加会让共同背景分量叠加，token 存在性反而模糊。因此不能把新 embedding
原样放进旧的加法子树状态。

## 结果二：Pair 排序改善，但路由坍缩

固定点实验使用同一份 20K/1024 token pair cache、seed 19301、300 steps：

| 初始化 | Embedding-only acc. | TreeHeap acc. | TH 增益 | 下游 leaf 使用 | LCP margin |
|---|---:|---:|---:|---:|---:|
| 随机 | `0.62090` | `0.62410` | `+0.00320` | `35/64` | `0.14980` |
| root-unfold | `0.62320` | `0.64350` | `+0.02030` | `1/64` | `0.00000` |

这里有一个真实正信号：root-unfold TreeHeap 比随机 TreeHeap 高 `0.01940`，
也比同起点 embedding-only 高 `0.02030`。这说明新坐标包含 pair 任务可用的
语料关系，旧实验的随机起点确实遮住了一部分信息。

但这不是层级路由成功。进入 fixed-point 以前，32 维 root 坐标使用
`31/32` 个维度，占用熵 `0.93336`；经过旧 TreeHeap 后却所有 token 都走到
同一个 6 层 leaf，正负样本 LCP 都等于 6。坍缩由旧 pair-NCE/READ 路由合同
引入，而不是 root-unfold embedding 先天坍缩。

## 结论

结论是“有改善，但接口不兼容”，不能写成简单成功或失败：

- **支持**：root-unfold embedding 含有可用于语料 pair 排序的信息；
- **否决**：它不能直接替换 compact route 中可加的随机近正交向量；
- **未修复**：旧 fixed-point 目标提高了 pair accuracy，却没有保住层级路由。

下一步不是盲目扩大词表或重复训练，而是明确两个新数学接口：

1. 概率坐标怎样组合成 parent，且保留“某 token 是否存在”的可辨识性；
2. READ/route 怎样在优化 pair 关系时保持输入坐标的层级结构。

任务日志为 `499..504`，正式数据位于同目录四个正式 arm，配对摘要见
`comparison.json`，原始 summary 哈希见 `SHA256SUMS`。
