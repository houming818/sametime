# SameTime

中文 | [English](README.md)

SameTime 是一个开放研究项目，正在探索一种称为 **TreeHeap** 的递归、固定容量
神经网络架构，用于语言编码、生成，以及 Encoder 与 Decoder 之间私有协议的形成。

项目由 **Houming818** 发起，并在人类与 AI 的持续协作中演化：数学想法先被写成
可证伪的 Claim，再实现为实验，由多个 Agent 交叉审核，成功与失败证据都保留下来。
因此，TreeHeap 不是一个突然公布的完成品；它如何一步步被发现，本身就是研究结果
的一部分。

## 核心问题

一个语言系统能否真正把信息组织成递归计算的 TreeHeap，而不是仍然保存一个
线性序列，只在外面套上树形索引？

当前候选数据流是：

```text
tokens
-> WRITE 到 leaf state
-> 固定容量 XOR Butterfly 通信
-> 有界、可逆的递归 FOLD
-> 多分辨率 H_state（root + parent/detail 各层 + leaves）
-> Decoder 递归 READ
-> token 概率分布
-> 自回归生成
```

其中：

- **FOLD** 把两个 child 合并成分辨率更粗的 parent，同时保留可恢复的 detail；
- **UNFOLD** 提供对应的代数重建路径；
- **Butterfly** 用稀疏局部 kernel 让远距离地址发生通信，同时不申请无界树空间；
- **H_state** 是运行时真正承载多种分辨率的 TreeHeap；
- **READ** 让 Decoder 使用整套层级状态，而不是把 root 当作句子 hash，或者从 leaf
  建立 flat 直通通道；
- 梯度学习负责在这些明确的结构约束中形成 Encoder 与 Decoder 的私有协议。

当前运行状态 `H_state` 是 TreeHeap；模型的学习参数仍然是由 TreeHeap kernel 共享
的张量集合，参数内存本身还没有被重构成可以生长的 TreeHeap。

## 为什么它不只是“画成树的数组”

早期实验暴露过多种错误捷径：flat `L x L` route、直接泄漏正确分支的几何 feature、
leaf-only Decoder、坍缩到最细层的 learned STOP，以及 Loss 持续下降、生成却只剩
固定句子的条件坍缩。这些失败都改变了后续架构，并且仍然保存在公开记录中。

只有在控制变量实验中，递归状态和树地址对输出产生可测量的因果影响，才算
TreeHeap 证据。仅仅把张量放进 heap 数组，不算使用了 TreeHeap。

## 当前研究状态

现有证据支持以下有限结论：

- TreeHeap 的递归算子与可逆 FOLD/UNFOLD 路径可以实现，并接受数值审计；
- TreeHeap 状态可以在真实 WMT seq2seq 数据上训练，并生成非空语言输出；
- Butterfly、递归 FOLD 和多层状态干预能够产生可测量的因果影响；
- Pretrain 与 Task Train 已能通过同一 checkpoint 和可复现数据管线连接。

仍未解决的问题包括：

- 让 coarse、middle、fine 分辨率形成稳定、互补的私有协议；
- 防止 Decoder 坍缩到某个最方便的单一深度；
- 在现有算力下获得稳定可用的长文本生成与更强翻译质量；
- 证明通用推理、对话、记忆或世界模型能力。

SameTime 目前不声称 TreeHeap 已经完成、具有意识或优于其他架构。它当前的贡献是：
提出了一个可以被实现、干预和证伪的新架构家族，并公开保存了这个架构被发现的
完整过程。

## 如何阅读项目

希望快速了解完整技术演化，可以从这里开始：

- [TreeHeap 生成航线技术演化地图](ara/s3-generation/EVOLUTION.zh.md)
- [ARA 论文式总览（中文）](ara/PAPER.zh.md)
- [ARA paper-style overview (English)](ara/PAPER.md)
- [当前 Claim 登记表](ara/s3-generation/logic/claims.md)

ARA 目录采用以下结构：

```text
logic/     Claim、Predict、实验合同与判决
src/       实现代码与评价程序
trace/     失败路径、路线变化及其理由
evidence/  机器可读摘要、日志、哈希和大文件指针
```

`C03`、`C12`、`C13` 等编号只是档案坐标，不是面向读者的能力等级。实验的人话名称、
中间改进与继承关系统一记录在演化地图中。

## 公开文章与复现

面向读者的 SPR / TreeHeap 系列文章：

- <https://www.grepcode.cn/spr/>
- <https://www.lostmap.cn/spr/>

第一个可以下载运行的 TreeHeap 翻译 POC 是历史版本
**STONE-1 Candidate C08**：

- <https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.tar.gz>
- <https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.sha256>
- <https://github.com/houming818/sametime/tree/stone1-candidate-c08>

它作为研究 checkpoint 保留，不代表 SameTime 的最终模型。大型 checkpoint、受许可
约束的语料和本地 NAS 文件不会直接提交到 Git；evidence 会保留复核所需的哈希、
摘要、命令和制品位置。

## 许可证

SameTime 的源代码和原创研究材料采用 GPL-3.0 发布，详见 [LICENSE](LICENSE)。
第三方数据集和模型继续遵守各自许可证，本仓库不会替它们重新授权或直接分发。
