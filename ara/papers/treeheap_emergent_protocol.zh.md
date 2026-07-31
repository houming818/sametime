# TreeHeap：可逆多分辨率树上的涌现式双语序列协议

**English title:** *TreeHeap: Emergent Bilingual Sequence Protocols on a Reversible Multiresolution Tree*

**状态：** 中文审阅稿 v0.1，2026-07-31

**作者：** Houming818（Independent Researcher）

**代码与证据：** SameTime / ARA，GPL-3.0

**结果边界：** 匹配消融实验已完成；全量双向训练仍在运行，文中明确标为中期观察。

## 摘要

本文研究一个存在性问题：除了线性序列和稠密矩阵之外，一个具有显式地址、递归层级和可逆分解的树形状态，能否仅通过序列到序列任务的梯度，自发形成一套编码与解码协议？我们提出 TreeHeap，一种固定容量、可寻址的多分辨率状态结构。输入 token 首先写入叶节点；稀疏 XOR-Butterfly 通信以共享的双节点可逆核在 `log2(N)` 个阶段连接全部叶地址；随后可逆 lifting FOLD 将细节逐层抽取到父节点，并保留精确 UNFOLD 所需的 detail 状态。递归 Decoder 不接收人工语法标签，而是根据自身隐状态，在 root、内部节点与 leaf 之间分配概率读取质量。整个系统只使用最终译文的 token 交叉熵训练。

在真实 WMT 中英数据上的匹配机制实验中，我们保持参数量、初始化、样本、批次顺序、优化器和 Decoder 一致，仅改变 TreeHeap 内部地址通信方式。三颗随机种子的平均测试 NLL 分别为：关闭通信 `4.65196`、重复相邻通信 `4.65415`、XOR-Butterfly 通信 `4.56509`；对应 token BLEU-4 为 `9.9501`、`9.9485` 和 `10.5462`。训练完成后关闭 Butterfly，平均 NLL 恶化 `1.16873`；把它替换成相邻通信也显著恶化。该结果不是模型排行榜比较，而是证明学习后的 TreeHeap 确实使用了 changing-bit 地址通信。

我们进一步启动一个单卡、1417 万平行句对、双向交替、最长 253 pieces 的规模化实验。第一轮训练尚未结束时，固定双向探针已经从高频重复发展为保留年份、事件和关系轮廓的自由译文；中期双向验证 NLL 为 `3.48393`，关闭 Butterfly 后为 `4.80849`。这些数据是正在生长的研究记录，不是最终质量结论。

本文不主张 TreeHeap 替代 Transformer、循环网络或其他架构。我们的结论更窄：可逆树形状态、稀疏地址通信和最终任务损失可以共同形成一种可训练的序列协议。不同架构拥有不同的归纳偏置，尺有所短，寸有所长。TreeHeap 的价值首先在于提供一种可以被复现、消融和继续研究的新算法对象。

## 1. 引言

现代机器学习已经证明，参数化函数可以通过梯度下降，从大量样本中形成复杂的输入输出映射。然而，“可以学习”并不意味着只能使用一种内存形状。序列、矩阵、图和树都可以承载参数与隐状态，它们对信息的组织方式不同。

TreeHeap 研究源自一个简单问题：如果模型状态本身是一棵有地址的树，而不是把树结构最终摊平成一段数组，那么梯度能否在这棵树上形成 Encoder 与 Decoder 共同理解的私有协议？这里的“私有协议”不是人工命名的主语、谓语或宾语，而是训练产生的内部编码。人类不必直接读懂每个内部节点，只需要验证：

1. 协议能够支持真实序列任务；
2. 结构操作在数学上明确并可逆；
3. 破坏结构会可测量地伤害结果；
4. 随着数据进入，固定探针的自由生成出现稳定改善；
5. 所有失败、边界和原始输出都能被复查。

本文沿着这五条要求组织，而不是围绕“击败某模型”组织。

### 1.1 研究问题

本文考察以下核心 Claim：

> 在不提供语法树、角色槽位或类比标签的情况下，一个共享参数的可逆 TreeHeap，能否依靠中英 Seq2Seq 交叉熵，自发形成双向编码、结构通信与递归读取协议？

该 Claim 被拆成三个可以证伪的子问题：

- **代数问题：** FOLD/UNFOLD 与通信是否保持闭合和可逆？
- **因果问题：** 学习结果是否真正依赖树上的地址通信？
- **生成问题：** 自由解码是否随数据规模从重复坍缩走向源相关输出？

### 1.2 贡献

本文的贡献不是宣布一种通用最优架构，而是：

1. 定义一个由可逆 lifting、detail 状态和递归概率 READ 组成的 TreeHeap Seq2Seq 系统；
2. 引入固定容量 XOR-Butterfly 通信，使任意叶地址可以通过 `log2(N)` 个稀疏阶段交换信息；
3. 设计参数完全匹配的内部消融，区分“增加参数”和“使用 changing-bit 地址拓扑”；
4. 在三颗随机种子的真实 WMT 实验中得到一致的机制证据；
5. 公开全量双向训练的 checkpoints、wake reports 与 `dreams` 生长轨迹，使中间失败也成为 evidence。

## 2. 研究立场：发现一种算法，而不是替代一种算法

不同数据结构对同一问题可能提供不同视角。矩阵擅长规则并行计算，序列天然表达时间顺序，图允许一般关系，树则提供前缀、层级、子结构和对数深度地址。一个架构在某类任务上的优势，不自动扩展为所有任务的优势。

因此本文遵守两个边界：

**第一，不进行跨架构优越性声明。** 相关工作用于说明数学来源和设计差异，不用于建立“谁取代谁”的叙事。

**第二，内部对照仍然必要。** 如果 Butterfly TreeHeap 优于同参数的 Identity TreeHeap，这只说明 Butterfly 组件对该 TreeHeap 有用；它不说明 TreeHeap 优于外部模型。内部消融是因果验证，不是排行榜。

## 3. TreeHeap 状态

### 3.1 地址与容量

给定最大叶容量 `N = 2^D`，TreeHeap 使用二叉堆地址：

```text
root = 0
left(i) = 2i + 1
right(i) = 2i + 2
```

在实现中，输入 token 首先形成叶状态：

$$
X^{(0)} = [x_0, x_1, \ldots, x_{N-1}], \qquad x_i \in \mathbb{R}^d.
$$

不足 `N` 的位置由 mask 关闭，不参与有效双节点通信。规模化版本的最大容量为 256，但短 batch 只展开到容纳自身所需的最近二次幂。32、64、128 和 256 叶共享同一组 kernel 参数，因此它们不是四个独立模型。

### 3.2 TreeHeap 不是只有 root

TreeHeap 完整状态记为：

$$
H = \left(r, \{d^{(0)}, d^{(1)}, \ldots, d^{(D-1)}\}, M\right),
$$

其中 `r` 是 root，`d^(k)` 是第 `k` 层 lifting detail，`M` 是各层有效地址 mask。root 提供最压缩的状态，但完整信息分布在 root 与不同分辨率的 detail 中。本文不假设 root 必须对应一条人类可读摘要。

## 4. 可逆地址通信

### 4.1 双节点耦合核

对一对状态 `(a,b)`，一个通信阶段定义为：

$$
b' = b + \alpha_s \tanh(F_\theta(a)),
$$

$$
a' = a + \alpha_s \tanh(G_\theta(b')).
$$

`F` 与 `G` 是所有地址和深度共享的小型 MLP kernel，`alpha_s` 是可学习的有界阶段增益。其逆运算按相反顺序计算：

$$
a = a' - \alpha_s \tanh(G_\theta(b')),
$$

$$
b = b' - \alpha_s \tanh(F_\theta(a)).
$$

因此，在浮点误差范围内，通信不要求丢弃输入状态。

### 4.2 XOR-Butterfly 调度

在阶段 `s`，地址 `i` 与地址

$$
j = i \oplus 2^s
$$

配对，其中 `oplus` 为按位 XOR。每个阶段包含 `N/2` 对局部操作，共有 `log2(N)` 个阶段，所以 pair 操作数量为：

$$
\frac{N}{2}\log_2 N = O(N\log N).
$$

经过全部阶段，每个地址都存在一条到任意其他地址的 changing-bit 路径。这里的 XOR 是明确的通信先验，不被解释成天然语义地址。语义是否利用这些路径，由任务梯度决定。

## 5. 可逆多分辨率 FOLD

### 5.1 Lifting 形式

对相邻左右状态 `(l,r)`，共享 Predictor `P` 与 Update `U` 定义：

$$
d = r - P(l),
$$

$$
p = l + U(d).
$$

`p` 进入上一层成为 parent，`d` 被保留为当前分辨率 detail。逆运算为：

$$
l = p - U(d),
$$

$$
r = d + P(l).
$$

递归执行后，`N` 个叶状态变为一个 root 和 `N-1` 个分布在各层的 detail 状态。由于逆式明确，模型不需要让 root 单独记住所有 token。

### 5.2 可学习 Update

实验中的 Update 以稳定的线性 lifting 为起点：

$$
U(d) = 0.5d + 0.5\tanh(U_\theta(d)).
$$

最后一层参数从零初始化，因此训练开始时系统等价于确定的 `0.5d` 更新；随后梯度可以改变 detail 对 parent 的贡献。

## 6. 递归概率 READ 与生成

Decoder 在每个输出时间步维护隐状态 `h_t`。对于深度 `k` 的候选节点 `n_i^(k)`，模型计算停止概率：

$$
p_{\text{stop}}(i,k,t)
= \sigma\left(S_\theta\left[q(h_t), n_i^{(k)} + e_k\right]\right),
$$

其中 `e_k` 是深度 embedding。未停止的概率质量在两个 child 之间分配：

$$
p(c\mid i,t) =
\operatorname{softmax}_c
\left(\frac{B_\theta(h_t)^\top n_c}{\sqrt d}\right).
$$

所有深度上停止的节点加权形成 context `c_t`。随后 GRU 状态更新并输出下一个 token 分布：

$$
h_{t+1}=\operatorname{GRU}([E(y_t),c_t],h_t),
$$

$$
p(y_{t+1})=\operatorname{softmax}(W_o[h_{t+1},c_t]).
$$

训练阶段使用 teacher forcing；唯一语言目标是目标 token 的交叉熵：

$$
\mathcal{L}_{\text{seq}}
= -\sum_t \log p_\theta(y_t\mid y_{<t},H_x).
$$

没有语法标签、角色分类损失或人工“正确内部节点”目标。内部协议由最终序列损失间接塑造。

## 7. 实验一：匹配 WMT 机制消融

### 7.1 目的

本实验不比较不同模型家族。它只回答：在同一个 TreeHeap 内，changing-bit 通信是否比“完全不通信”或“只重复最近邻通信”提供了可学习的结构作用？

### 7.2 三个实验臂

| 实验臂 | 通信调度 | 解释 |
|---|---|---|
| Identity | 分配相同 kernel，但跳过 pair update | TreeHeap 原始基线 |
| Adjacent | 每个阶段都重复 bit-0 相邻配对 | 控制额外深度和参数 |
| Butterfly | 阶段 `s` 使用 `i XOR 2^s` | changing-bit 稀疏通信 |

三个实验臂具有完全相同的参数数量 `34,445,832`。通信 kernel 的输出层从零初始化，因此三个臂从相同函数起点出发。样本、初始化、batch 顺序、优化器、训练轮数、Decoder 与测试集全部匹配。

### 7.3 数据与训练

语料为本地 WMT massive 中英 TSV。每颗种子从前 200 万行进行确定性 reservoir sampling，选择：

```text
train / valid / test = 200,000 / 5,000 / 5,000
source/target length = 8..32 SentencePiece pieces
direction = English -> Chinese
epochs = 5
dim / hidden = 256 / 256
seeds = 8104, 8105, 8106
```

训练运行于单张 RTX 3090。生成指标是项目代码中固定实现的 token BLEU-4，不与外部排行榜数值直接比较。

### 7.4 预注册判定

Butterfly 必须在至少两颗种子中同时满足：

1. 测试 NLL 优于 Identity 至少 `0.02`；
2. 优于 Adjacent 至少 `0.015`；
3. 长源子集收益不消失；
4. 训练后关闭通信至少恶化 `0.02` NLL；
5. source shuffle 至少恶化 `0.50` NLL；
6. 多个 TreeHeap 分辨率被 READ 使用；
7. 梯度与生成有限且非空。

## 8. 实验一结果

### 8.1 测试损失

| Seed | Identity NLL | Adjacent NLL | Butterfly NLL | Butterfly 相对 Identity | 相对 Adjacent |
|---:|---:|---:|---:|---:|---:|
| 8104 | 4.62285 | 4.62273 | **4.54546** | 0.07738 | 0.07727 |
| 8105 | 4.66175 | 4.67974 | **4.58915** | 0.07260 | 0.09059 |
| 8106 | 4.67127 | 4.65998 | **4.56066** | 0.11061 | 0.09932 |
| Mean | 4.65196 | 4.65415 | **4.56509** | **0.08687** | **0.08906** |

平均 token BLEU-4 为：

| Identity | Adjacent | Butterfly |
|---:|---:|---:|
| 9.9501 | 9.9485 | **10.5462** |

三个实验种子全部通过预注册 gate。该表只说明 changing-bit 通信对这个 TreeHeap 有效，不构成跨架构结论。

### 8.2 长源与因果干预

在 25--32-piece 长源子集上，Butterfly 相对 Identity 的 NLL 收益分别为：

```text
0.07451 / 0.07825 / 0.10424
mean = 0.08567
```

训练完成后进行运行时结构替换：

| 干预 | 平均结果 |
|---|---:|
| Butterfly 改为 Identity | NLL 恶化 1.16873 |
| Butterfly 改为 Adjacent | NLL 约恶化 1.06183 |
| Source shuffle | NLL 约恶化 3.15625 |

如果收益只来自“多了几层 MLP”，把通信改为相同深度的 Adjacent 不应产生如此显著损伤。结果支持更窄的因果解释：模型参数适应并使用了原生 changing-bit 调度。

### 8.3 数学完整性

三颗种子中：

```text
Butterfly forward/inverse MSE：约 6.7e-16 .. 7.2e-16
FOLD/UNFOLD closure MSE：约 6.7e-15 .. 8.2e-15
dense N x N attention allocation：无
```

可逆误差接近 FP32 数值精度，说明任务训练没有破坏代数闭合。

## 9. 实验二：全量双向生长观察

### 9.1 设计

本实验正在运行，尚不提供最终 Claim。原始语料为：

```text
14,170,275 parallel rows
2.4 GiB TSV
maximum content = 253 pieces
maximum TreeHeap = 256 leaves
```

每条平行语料在相邻 epoch 中交换方向。方向由显式 `en2zh` 或 `zh2en` token 指示，但 Encoder、Butterfly、FOLD/UNFOLD、READ 和 Decoder 参数全部共享。训练采用流式 50K 行 block，并在 block 边界原子保存 checkpoint。

### 9.2 Dreams 观察协议

固定文件 `dreams.txt` 保存人类选择、从不参与训练的双向句子。每约 100 万样本，当前 checkpoint 对这些句子自由解码，并生成不可变快照。该设计不把主观样例当作损失，但允许观察同一输入随训练发生的变化。

在约 3 万样本时：

```text
SOURCE: The new company is expected to begin operations in the spring of 2019.
DREAM:  新新是是是是是是是是是是是是是是是是是……
```

在约 1198 万样本时：

```text
SOURCE: The new company is expected to begin operations in the spring of 2019.
DREAM:  新年公司预计于2019年,预计在2019年春天开始运营。
```

反方向：

```text
SOURCE: 新公司预计将在2019年春季开始运营。
DREAM:  New Year's anticipation will start to operate in spring and 2019.
```

输出仍有实体偏差、重复和不自然表达，但已经由源无关高频坍缩发展为保留年份、事件和方向的源相关序列。

### 9.3 中期数字

在 `11,975,676` 个训练样本、`253,253,198` 个目标 token 时：

| 指标 | 数值 |
|---|---:|
| en2zh validation NLL | 3.78795 |
| zh2en validation NLL | 3.17990 |
| mean validation NLL | 3.48393 |
| runtime Identity NLL | 4.80849 |
| Identity damage | +1.32457 |

这些数字不能与实验一直接作严格增量比较，因为方向、长度分布、词表和数据划分均已改变。它们仅描述同一次规模训练内部的生长状态。

## 10. 讨论

### 10.1 什么是“涌现协议”

本文使用“涌现”一词时，不表示不可解释的神秘现象。它有一个操作性定义：

1. 训练目标只评价最终 token；
2. 没有标签规定内部节点应该存什么；
3. Encoder 和 Decoder 共同适应一套内部表示；
4. 结构干预会破坏这套表示；
5. 自由输出随数据形成源相关规律。

因此，“私有”意味着协议不是人工命名的，不意味着它无法被实验观察。

### 10.2 root 与 detail

实验并未证明“越靠近 root 就一定是人类语言中的摘要”。更准确的说法是：TreeHeap 提供多个递归分辨率，Decoder 学习在不同时间步如何分配读取质量。哪个深度承载何种信息，是任务与参数共同决定的。

### 10.3 为什么需要 Butterfly

纯二叉相邻 FOLD 使远距离叶节点只能经过多层 parent 间接相遇。Butterfly 在 FOLD 前提供固定深度的全地址稀疏通信。它没有取消树，而是为树的叶平面增加一组可逆“神经纤维”，随后信息仍进入 lifting 的 root/detail 分解。

### 10.4 尺有所短，寸有所长

TreeHeap 的显式层级与可逆 detail 便于进行地址干预、深度观察和子结构实验；代价是当前递归 Decoder 仍需展开层级并对节点进行软读取。本文实现的生成复杂度并未被证明优于其他架构，也尚未实现真正的压缩存储收益。

算法多样性本身具有研究价值。一个新架构首先应该证明自己能够稳定存在，然后才有资格讨论它适合哪些任务。

## 11. 局限与风险

1. **翻译质量仍处于研究 PoC。** 实验一 BLEU-4 约 10.55，不能作为商业翻译器使用。
2. **全量训练尚未完成。** 实验二所有数字均为中期快照。
3. **没有外部模型优越性结论。** 本文只做 TreeHeap 内部机制消融。
4. **Decoder 计算仍然昂贵。** 当前实现展开总计小于 `2N` 个层级节点，每个输出步进行软 READ，生成大致为 `O(TN)`。
5. **地址语义未知。** XOR 地址是通信调度，不等于已经发现语义坐标系。
6. **数据存在噪声。** WMT massive 含网页、商品、成人语句和实体错配；模型输出会继承这些问题。
7. **评估尚不完整。** 最终稿需要补充标准 BLEU、chrF、COMET、重复率、长度分桶和人工盲评。
8. **硬件范围有限。** 当前主要证据来自单张 RTX 3090；跨硬件和多卡可扩展性尚未验证。

## 12. 可复现性

核心文件：

```text
ara/s3-generation/src/s2_treeheap_butterfly_wmt.py
ara/s3-generation/src/s3_treeheap_butterfly_bilingual_full.py
ara/s3-generation/logic/treeheap_butterfly_wmt_ablation.md
ara/s3-generation/logic/treeheap_butterfly_bilingual_full_train.md
```

正式三种子 evidence：

```text
ara/s3-generation/evidence/s2_treeheap_butterfly_wmt_formal/
```

关键 Git 记录：

```text
373c0b4  preregistration
8f48cbe  formal evidence and supported decision
2eef49e  bilingual full trainer
a0feb5b  bilingual smoke evidence
```

约 132 MiB 的正式 checkpoints 保留在 `io`，其 SHA-256 已记录于 evidence 的 `CHECKPOINTS.md`。代码使用 GPL-3.0 发布。

## 13. 结论

本文没有回答“哪一种 AI 架构最好”，也不试图回答。我们完成的是一个更基础的存在性证明：固定容量的树形状态可以在严格可逆的 FOLD/UNFOLD 代数上，通过稀疏 changing-bit 通信和最终 Seq2Seq 梯度，形成可测量、可干预的语言协议。

三种子匹配实验说明 Butterfly 通信不是闲置装饰；关闭或替换它会显著破坏已学习模型。全量双向实验进一步显示，固定探针可以从重复坍缩逐渐长出源相关的翻译轮廓。协议仍不成熟，但它已经可以被训练、观察、破坏、恢复和继续生长。

这正是本文希望建立的起点：不是替代已有智能，而是为智能提供另一种可以栖居的数据结构。

## 参考文献

1. Vaswani, A., et al. *Attention Is All You Need*. NeurIPS, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
2. Tai, K. S., Socher, R., and Manning, C. D. *Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks*. ACL, 2015. [arXiv:1503.00075](https://arxiv.org/abs/1503.00075).
3. Sweldens, W. *The Lifting Scheme: A Custom-Design Construction of Biorthogonal Wavelets*. Applied and Computational Harmonic Analysis, 1996.
4. Gomez, A. N., Ren, M., Urtasun, R., and Grosse, R. B. *The Reversible Residual Network: Backpropagation Without Storing Activations*. NeurIPS, 2017. [arXiv:1707.04585](https://arxiv.org/abs/1707.04585).
5. Dao, T., Gu, A., Eichhorn, M., Rudra, A., and Ré, C. *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*. ICML, 2019. [arXiv:1903.05895](https://arxiv.org/abs/1903.05895).
6. Ebrahimi-Fard, K., and Rahm, L. *A Survey on the Munthe-Kaas-Wright Hopf Algebra*. 2023. [arXiv:2306.04381](https://arxiv.org/abs/2306.04381).

## 附录 A：Claim 边界表

| 陈述 | 当前状态 |
|---|---|
| TreeHeap FOLD/UNFOLD 在训练后保持数值可逆 | 支持 |
| XOR-Butterfly 在固定容量中提供全地址稀疏路径 | 数学定义成立 |
| 模型在真实 WMT 中使用 Butterfly 通信 | 三种子支持 |
| 一个共享 TreeHeap 能开始形成中英双向协议 | smoke 与中期结果支持 |
| root 是人类可读摘要 | 未证明 |
| XOR 地址天然具有语义 | 未证明 |
| TreeHeap 比其他架构更好 | 未声明，也未证明 |
| 当前模型达到产品级翻译质量 | 否 |
| 当前实现节省训练或生成计算 | 未证明 |

## 附录 B：最终稿待补实验

1. 完成 96 小时双向训练并冻结最终 checkpoint；
2. 在固定测试集报告标准 BLEU、chrF、COMET；
3. 报告 8--32、33--64、65--128、129--253 长度区间；
4. 报告自由生成重复率、空输出率与 unique-output rate；
5. 对最终 checkpoint 重做 Identity、Adjacent、source shuffle；
6. 对 `dreams` 进行时间序列分析，但不把探针选择成训练目标；
7. 增加独立人工盲评与数据污染审计；
8. 测量实际 wall-clock、GPU hours、token throughput 与显存。
