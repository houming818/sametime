# TreeHeap：从树形直觉到可逆多分辨率双语协议

**English title:** *TreeHeap: From Tree-Structured Intuition to a Reversible Multiresolution Bilingual Protocol*

**状态：** 中文审阅稿 v0.4，2026-08-01

**作者：** Houming818（Independent Researcher）

**研究设计：** Houming818 与 Codex（Review Engineer）协作完成

**审计协作：** DeepSeek、GLM

**代码与证据：** SameTime / ARA，GPL-3.0

**结果边界：** 三臂匹配训练比较已完成；严格的同 checkpoint 拓扑消融仍待补做；全量双向训练仍在运行，文中明确标为中期观察。

## 摘要

本文记录 TreeHeap 算法从直觉、争论、失败实验到真实双向生成的形成过程。最初的设想不是先规定一种语法树，而是把模型状态放进一个固定容量、有地址、有递归深度的树堆：局部 kernel 在子堆上卷积，信息逐层向 root 抽取，Decoder 再沿着同一结构展开和读取。算法必须在有限内存中生长，必须允许数学上的 FOLD/UNFOLD，且内部编码可以由任务梯度自行形成。围绕这些约束，Houming818 提供关于秩序、旋转、抽水机、分辨率和私有协议的原始直觉；Codex 将其转写为代数定义、Claim、Predict、Proof 和可复现实验；DeepSeek 与 GLM 对泄漏、错误实现和过强结论进行交叉审计。

最终形成的 TreeHeap 是一种固定容量、可寻址的多分辨率状态结构。输入 token 写入叶节点；稀疏 XOR-Butterfly 通信以共享的双节点可逆核在 `log2(N)` 个阶段连接全部叶地址；可逆 lifting FOLD 将细节逐层抽取到父节点，同时保留精确 UNFOLD 所需的 detail 状态。递归 Decoder 根据自身隐状态，在 root、内部节点与 leaf 之间分配概率读取质量。系统不接收人工语法结构，只使用最终译文的 token 交叉熵训练。

在真实 WMT 中英数据上的匹配训练实验中，我们保持参数量、初始化、样本、批次顺序、优化器和 Decoder 一致，仅改变 TreeHeap 内部地址通信调度。三颗随机种子的平均测试 NLL 分别为：关闭通信 `4.65196`、重复相邻通信 `4.65415`、XOR-Butterfly 通信 `4.56509`；对应 token BLEU-4 为 `9.9501`、`9.9485` 和 `10.5462`。这支持一个有限结论：在该训练合同下，Butterfly 配置比两个对照配置得到更低损失。训练完成后直接旁路 Butterfly 会使 NLL 恶化，但该操作同时移除了整个已学习变换并改变后续模块接收的坐标分布，因此只能作为模块依赖检查，不能单独证明 changing-bit 拓扑具有因果优势。

我们进一步启动一个单卡、1417 万平行句对、双向交替、最长 253 pieces 的规模化实验。第一轮训练尚未结束时，固定双向探针已经从高频重复发展为保留年份、事件和关系轮廓的自由译文；中期双向验证 NLL 为 `3.48393`。该实验曾报告运行时 Identity NLL，但复核发现 native 与 Identity 使用的验证样本数量不一致，所以该差值现已撤出正式证据，只保留为需要重算的诊断记录。这些数据是正在生长的研究记录，不是最终质量结论。

这些结果说明：可逆树形状态、稀疏地址通信和最终任务损失可以共同形成一种可训练的序列协议。TreeHeap 还很年轻，但它已经从概念发展为可以被复现、比较、干预、恢复和继续训练的算法对象；关于 Butterfly 拓扑本身的严格因果结论仍保持开放。

## 1. 引言

TreeHeap 研究源自一个关于内存形状的问题：如果模型状态本身是一棵有地址的树，梯度能否在树的 root、路径、内部节点和 leaf 之间形成 Encoder 与 Decoder 共同理解的私有协议？

最初的 TreeHeap 被理解为一个“神经元单侧切片”：一棵有序树描述一个局部方向上的状态，有限旋转可以提供其他观察方向，多个切片在固定容量中组合成更完整的状态。这个想法首先提供的是几何直觉，而不是现成公式。研究很快遇到四个必须回答的问题：

1. 参数究竟存在哪里，是 kernel 的 `theta`，还是每个样本的 TreeHeap 状态 `H`？
2. 多个 token 怎样真正合成一棵树，而不是换一种方式存放数组？
3. 信息怎样从 leaf 逐层进入 parent，同时又能被 Decoder 取回？
4. 长距离 leaf 如何相互作用，而不无限申请新节点或把整棵树摊平？

我们没有一次性回答这些问题，而是在数十个 Claim 和失败实验中逐步缩小设计空间。这里的“私有协议”最终被定义为训练产生的内部编码，而不是人工命名的主语、谓语或宾语。人类不必直接读懂每个内部节点，只需要验证：

1. 协议能够支持真实序列任务；
2. 结构操作在数学上明确并可逆；
3. 破坏结构会可测量地伤害结果；
4. 随着数据进入，固定探针的自由生成出现稳定改善；
5. 所有失败、边界和原始输出都能被复查。

本文沿着设计问题、失败证据、数学修正和最终效果组织。

### 1.1 研究问题

本文考察以下核心 Claim：

> 在不提供语法树、角色槽位或类比标签的情况下，一个共享参数的可逆 TreeHeap，能否依靠中英 Seq2Seq 交叉熵，自发形成双向编码、结构通信与递归读取协议？

该 Claim 被拆成三个可以证伪的子问题：

- **代数问题：** FOLD/UNFOLD 与通信是否保持闭合和可逆？
- **因果问题：** 学习结果是否真正依赖树上的地址通信？
- **生成问题：** 自由解码是否随数据规模从重复坍缩走向源相关输出？

### 1.2 贡献

本文的贡献是：

1. 定义一个由可逆 lifting、detail 状态和递归概率 READ 组成的 TreeHeap Seq2Seq 系统；
2. 引入固定容量 XOR-Butterfly 通信，使任意叶地址可以通过 `log2(N)` 个稀疏阶段交换信息；
3. 设计参数匹配的三臂训练比较，并明确区分“架构训练收益”“已训练模块依赖”和“changing-bit 拓扑因果性”三个不同问题；
4. 在三颗随机种子的真实 WMT 实验中得到一致的架构比较证据；
5. 公开全量双向训练的 checkpoints、wake reports 与 `dreams` 生长轨迹，使中间失败也成为 evidence。

## 2. TreeHeap 是怎样商量出来的

TreeHeap 不是从一张完成的架构图开始的。它由人类直觉、数学翻译、代码实验和独立审计反复往返形成。本节记录对最终算法真正产生影响的设计转折。

### 2.1 第一阶段：路径不是语义

我们最早拥有的 TreeHeap 不是现在的可逆多分辨率状态，而是一种由 token ID 决定路径的树形索引。它看起来已经具备地址、前缀和子堆，因此最自然的第一步，是问这些路径能否直接承担语言结构。

#### 2.1.1 路径邻接与前缀相似

第一种尝试把路径距离当成结构距离。例如两个 token 的路径拥有较长公共前缀，就认为它们可能属于同一个短语；在堆地址上相邻，就提高彼此连接的权重。这个想法利用了树的真实性质：路径确实能表示“一个节点在哪里”，公共前缀也确实能表示“两个地址何时分叉”。

问题在于，早期路径由 token ID 产生。同一个 `cat` 无论在句子中是施事、受事还是被修饰对象，基础路径都相同；两个 ID 接近的 token 也不必具有句法关系。实验 C-023 最终确认，路径更接近词表地址，而不是当前句子的语法角色。

我们由此第一次分开了两个此前混合的概念：

```text
路径回答：状态放在哪里？
语义回答：这个状态在当前上下文中表示什么？
```

TreeHeap 仍然需要路径，但不能把地址本身解释为语义。

#### 2.1.2 位权、角色基与外积张量

Houming818 随后提出“321 与 123”的位权类比。数字之所以因排列不同而改变，不只是因为包含 `1、2、3`，而是因为每个数字乘上了百位、十位和个位这些不同的基。对应到句子，一个 token 向量 `s_i` 也许需要乘上一个结构角色基 `e_role`：

$$
T_{sentence} = \sum_i s_i \otimes e_{role(i)}.
$$

这样，`cat` 放入 SUBJECT 槽与放入 OBJECT 槽会形成不同张量。我们测试了 one-hot、随机和正交角色基，也测试了有序拼接、外积和其他非交换组合。结果确认了一件事：这些算子可以让不同排列得到不同表示。也就是说，它们具有**表达顺序差异的容量**。

但这里存在一个循环：若构造张量前已经知道哪个 token 是 SUBJECT，结构问题其实已经由外部答案解决。随机角色基能区分两个排列，也不表示正确排列会自动获得更低能量。实验 C-022 中，候选排列的能量范围约为 `0.06`，说明算子对排列敏感；可是 gold 排列并不稳定处于最低能量。

这一步把我们的理解从：

```text
不同结构能否被表示？
```

推进为更严格的问题：

```text
任务数据能否通过梯度，让正确结构在这些表示中获得可选择的优势？
```

“能够区分”只是表示能力，“知道选谁”才是学习协议。

#### 2.1.3 从人工语法角色转向 latent slot

角色基实验失败后，我们没有直接放弃 slot，而是改变了 slot 的含义。ROOT、SUBJECT、OBJECT 是人类预先定义的语法标签；TreeHeap 真正需要的也许是训练后自行形成的 latent slot。它们可以被临时编号为 slot-0、slot-1、slot-2，却不预先规定对应什么语言学名称。

为判断这种结构是否在容量上可行，我们统计了真实样本中的 FoldNode 子节点数量。12K WMT 审计显示，四个 slot 可以覆盖约 `99.0%` 的节点，五个 slot 可以覆盖约 `99.8%`。这支持“小型槽位预算足以容纳多数局部组合”的结构判断，却没有证明这些 slot 已经自行学出语义。

同一阶段还尝试了概率容器。与其过早把一个节点固定挂到唯一 parent，不如保留多个候选：

```text
Parent A: 0.62
Parent B: 0.25
Parent C: 0.13
```

在当时的 parent 候选实验中，gold parent 的 top-1、top-3、top-5 覆盖率分别为 `93.1%`、`99.9%`、`100%`。这说明延迟坍缩可以保留几乎全部正确候选。但它只证明“候选桶没有过早丢掉答案”，尚未证明后续模块能利用这些概率完成生成。后来递归 READ 的 `{stop,left,right}` 概率质量，继承了“信息不足时先保留分布”的思想，但读取对象和训练合同已经重新设计。

#### 2.1.4 `t_merge` 与世界模型背景场

接下来我们尝试让语义向量携带上下文。早期表述是 `L0 × 背景场`：L0 token 向量与一个上下文场通过 CMul 等操作结合，再经过 `t_merge` 形成句子状态。Houming818 后来把“背景场”统一称为“世界模型”，希望它提供类似以下关系的参考系：

```text
ball + foot -> football
ball + hand -> basketball
```

如果参考系成立，那么 `football - ball` 不应只得到一个任意向量，而应指向 foot、kick、field、goal 一类局部关系方向。为此，我们比较了 L0、path、CMul pre-merge、merge-no-bias、最终 tree 以及 centered-tree 等多个读出点。

诊断得到的结果比“`t_merge` 直接毁掉空间”更复杂：最终向量受到一个很强的公共方向支配；去中心化或在 merge 前读取时，部分差异仍存在。这说明信息不一定在 `t_merge` 一步完全消失，也可能是 checkpoint、共同偏置和训练 loss 没有建立所需参考系。随后进行的局部上下文训练仍未形成稳定的 relation-anchor 排名，世界模型 Claim 因此被标记为 inconclusive，而不是 supported。

这次失败很重要。它让我们认识到，不能因为向量之间存在距离，就宣布拓扑世界模型已经形成。一个世界参考系至少需要满足：同类关系在不同词组上产生可迁移方向，正向 anchor 排名稳定优于困难负例，并且结果能在新训练 checkpoint 上复现。

#### 2.1.5 旧 checkpoint 审计与第一次大退航

最后，我们把上述设想放进一次统一的 12K WMT 策略审计：同时比较 random、L0 和 TreeHeap 向量，one-hot、random 和 orthogonal 角色基，多种非交换张量、FoldNode 度数以及 parent 概率容器。

审计发现，历史三轮训练 checkpoint 的 TreeHeap 向量两两平均 cosine 为 `0.9849`。它们几乎指向同一方向，张量 margin 接近零，也没有在角色槽模板排序上稳定超过 L0 或随机向量。这意味着旧 checkpoint 不能为“正确句子天然具有最低 TreeHeap 能量”背书。

我们没有把这解释成 TreeHeap 已被否定。被否定的是一个更具体的假设：**只要把 token 放进树路径，再做一次向量合并，语义结构就会自然出现。** 从这里开始，项目暂时退出句法能量搜索，退回 M0 数学层，先问 TreeHeap 对象本身支持哪些闭合算子、信息怎样可逆地进入 parent、梯度又怎样穿过这些算子。

第一阶段最终留下四条边界：

```text
地址可以承载结构，但地址本身不是语义。
排列可区分，不等于正确排列可选择。
概率容器可以延迟丢失，但不能弥补空洞的状态。
世界模型必须由可迁移关系证据建立，不能由向量距离命名出来。
```

### 2.2 第二阶段：先建立可以闭合的算子

Houming818 提出，TreeHeap 不应只是一种容器；它需要自己的 `Zero`、`plus`、`diff`、compose、decompose、mirror 和子堆 kernel。Codex 将这些要求转写为 M0 数学 probe，区分两类推理：

- 由定义直接成立的演绎性质，例如确定性 mirror 和精确 compose/decompose；
- 必须从数据归纳得到的概率性质，例如 kernel 参数和 READ 分布。

这一步的重要收获不是找到了一条语言规则，而是建立了研究纪律：确定性算子用闭包和逆运算证明；可学习算子用 loss、梯度、对照与干预验证。后来采用的 lifting FOLD/UNFOLD 正是沿着这条要求产生的。

### 2.3 第三阶段：从“加和 parent”到信息抽水机

最简单的 parent 定义是对子节点向量求和或平均。它能压缩形状，却会丢失左右关系和子堆身份。紧凑 route 实验明确失败：随机 token 向量加和后，Decoder 无法稳定恢复目标。

Houming818 用“抽水机”描述希望看到的过程：

```text
leaf 保存具体 token 和局部细节；
parent 接收子节点共同形成的信息；
越向 root，分辨率越粗；
Decoder 需要细节时，必须沿同一套管道取回。
```

一个关键修正是：抽水不能只是不可逆的平均。若 parent 只保存平均值，丢失的信息无法区分是合理退火还是实现损坏。于是我们采用 lifting：parent 保存更新后的 anchor，detail 保存预测残差。FOLD 可以改变分辨率，UNFOLD 又可以数值恢复完整状态。

### 2.4 第四阶段：root 不是全部意识状态

早期 Decoder 多次选择只从 root 读取，因为这是最短的优化路径。root-exclusive 实验虽然让 root 变得更有因果性，却损害了翻译 NLL。这说明强迫 root 单独恢复句子并不符合完整 TreeHeap 状态。

最终我们把样本状态定义为：

```text
H = root + all lifting details + masks
```

Decoder 的任务不是“从 root 猜回一切”，而是在每个生成时间步，根据当前隐状态决定在哪个深度停止、向哪个 child 继续。root、内部节点和 leaf 都可以参与，读取比例由任务 loss 学习。

### 2.5 第五阶段：修复长距离 leaf 通信

二叉 FOLD 的局部性带来新问题：相距很远的 leaf 要经过多层 parent 才能相遇。我们讨论过循环位移、矩阵视觉层和扩大邻接窗口，但这些方案要么覆盖不完整，要么缺少可逆性。

最终采用的 XOR-Butterfly 来自三个共同约束：

1. 固定容量，不允许递归复制导致内存指数生长；
2. 只使用共享的局部双节点 kernel；
3. 在有限阶段内让所有地址存在通信路径。

阶段 `s` 把地址 `i` 与 `i XOR 2^s` 配对。它把“旋转/换观察方向”的几何直觉，收敛成一个有限、可逆、可以逐阶段审计的地址调度。

### 2.6 第六阶段：让训练过程可以被看见

单个 NLL 无法告诉人类自由生成究竟发生了什么。我们建立 `dreams.txt`：其中保存固定的中英双向句子，永不进入训练。每约 100 万样本，当前 checkpoint 对这些句子自由解码并保存新文件。

因此，研究者既能看到损失曲线，也能看到同一个 TreeHeap 如何从：

```text
新新是是是是是……
```

逐渐发展为：

```text
新年公司预计于2019年，预计在2019年春天开始运营。
```

Dreams 不是评价函数，也不会反向指导梯度。它是一扇固定观察窗。

### 2.7 协作方法

整个过程采用 ARA 工作流：

```text
直觉 -> Claim -> Predict -> Experiment -> Evidence -> Audit -> Revision
```

Houming818 负责提出结构直觉、反驳不符合 TreeHeap 精神的实现，并决定研究目标；Codex 负责把直觉组织为数学、代码和可证伪实验；DeepSeek 与 GLM 负责独立代码审计。route feature 泄漏、flat 路由伪装成树、错误 CLI 标签和过强博客结论，都曾被审计发现并公开降级。

这种协作没有消除错误。它的作用是让错误能够留下证据，并成为下一版算法的输入。

### 2.8 失败怎样改变了算法

TreeHeap 的主要设计变化不是来自一次灵感完成，而是来自失败后的约束收紧。

| 阶段 | 当时的实现或假设 | 观察到的问题 | 进入下一版的修正 |
|---|---|---|---|
| 路径语义 | 用 token ID 路径直接解释句法 | 路径只保证地址唯一，不保证当前语义角色 | 把地址与任务学习分开 |
| 张量能量 | 用随机角色基区分排列 | 能区分两个排列，不等于正确排列自然具有最低能量 | 不把可区分性写成生成证据 |
| 历史 TreeHeap 向量 | 用旧 checkpoint 为结构能量背书 | 平均 cosine 约 `0.985`，表示方向高度坍缩 | 降级旧结论，重新训练和审计 |
| flat route | 每个长度学习一张路由表 | 只记住训练长度，树地址没有真正参与 | 改为共享的递归 kernel |
| geometry route | kernel 输入目标所在左右区间的布尔值 | 答案被直接写进 feature，得到虚假的 `1.0` accuracy | 删除泄漏，只允许读取 query 与节点状态 |
| 紧凑加和 state | 子节点随机向量直接相加 | 子堆身份和左右次序丢失 | 引入可逆 detail，不再只保留和 |
| root-exclusive | 强迫 Decoder 只从 root 读 | root 因果性提高，但序列 NLL 恶化 | 把 `H` 定义为 root、全部 detail 与 mask |
| 固定 lifting | `parent = left + 0.5 * detail` | 数学稳定，但信息价值分配没有适应任务 | 在固定起点上增加可学习 Update |
| 局部 FOLD | 只让相邻 leaf 逐层相遇 | 长距离地址要经过较多层才能交互 | FOLD 前加入固定容量 Butterfly 通信 |
| 全量生成 | 只看平均 NLL | 条件坍缩和高频重复可能被均值掩盖 | 增加固定 `dreams`、重复率和运行时消融 |

这张表也规定了论文的证据标准：一个模块只有在正确实现、无输入泄漏、通过对照并在干预后造成可测损伤时，才被认为参与了任务。

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

### 3.3 学习参数 `theta` 与样本状态 `H(x)`

这两个对象必须严格区分。

**学习参数 `theta`** 保存在 checkpoint 中，对全部训练样本共享。当前实现包括：

$$
\theta = \{E_{src}, E_{tgt}, F, G, \alpha, P, U, E_{depth}, S, B, \operatorname{GRU}, W_o\}.
$$

- `E_src`：把输入 piece 写成叶向量的 embedding；
- `E_tgt`：Decoder 已生成 piece 的 embedding；
- `F,G,alpha`：Butterfly 双节点通信 kernel 及各阶段增益；
- `P,U`：lifting 的 Predictor 与 Update；
- `E_depth,S,B`：深度表示、stop kernel 与左右 branch kernel；
- `GRU,W_o`：生成历史状态与输出词表分布。

**样本状态 `H(x)`** 是输入句子 `x` 经过这些共享参数计算后形成的临时 TreeHeap：

$$
H_\theta(x) = \left(r_x, \{d_x^{(k)}\}_{k=0}^{D-1}, M_x\right).
$$

`H(x)` 随输入变化，不作为一份新的模型参数永久保存。训练真正做的是调整 `theta`，使不同输入形成对任务有用的 `H(x)`。因此，`theta` 是形成私有协议的长期规律，`H(x)` 是协议对当前句子的实例化状态。

### 3.4 完整前向过程

对一个输入序列 `x`，TreeHeap Encoder 按以下顺序工作：

```text
ENCODE(x):
  1. WRITE
     把 direction、输入 pieces 和 EOS 写入有效 leaf；其余地址由 mask 关闭。

  2. COMMUNICATE
     对 s = 0 .. log2(N)-1：
       令地址 i 与 i XOR 2^s 成对；
       用共享可逆 kernel 更新这一对状态。

  3. FOLD
     从 leaf 层开始，递归处理每对 left/right：
       detail = right - P(left)
       parent = left + U(detail)
     保存每层 detail，直到只剩 root。

  4. RETURN
     返回 H(x) = (root, all_details, masks)。
```

Decoder 不把 `H(x)` 还原成原输入字符串再生成。它先用 UNFOLD 取得所有可读分辨率，然后在每个输出时间步递归分配读取概率：

```text
DECODE(H):
  levels = UNFOLD(root, all_details, masks)
  hidden = 0
  previous = BOS

  对 t = 0 .. T-1：
    从 root 开始；
    对当前可达节点计算 stop 概率；
    未停止的概率按 branch 分数进入 left/right child；
    汇总所有停止节点，得到 context_t；
    hidden_t = GRU(previous, context_t, hidden_(t-1))；
    输出词表概率 p(y_t)；
    训练时使用真实 previous，生成时使用模型上一步输出。
```

最后的交叉熵梯度依次穿过输出层、递归 READ、UNFOLD、root/detail、lifting 与 Butterfly，回到 token embedding 和全部共享 kernel。Encoder 与 Decoder 没有一张预先商定的“食物表”或“语法表”；它们能共同使用的内部编码，只能在这条端到端梯度链中形成。

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

## 7. 实验一：匹配 WMT 架构比较与依赖检查

### 7.1 目的

本实验包含两个必须分开的研究问题：

1. **从头训练的架构比较：** 在相同训练合同下，changing-bit、完全不通信和重复最近邻通信最终得到的任务损失是否不同？
2. **训练后的依赖检查：** 已经用 Butterfly 训练的模型，在运行时移除或替换通信后是否受损？

第一个问题比较三种训练配置；第二个问题只说明已训练系统是否依赖其内部变换。第二个问题不能单独回答 Butterfly 拓扑是否优于其他同容量拓扑。

### 7.2 三个实验臂

| 实验臂 | 通信调度 | 解释 |
|---|---|---|
| Identity | 分配相同 kernel 参数，但训练和推理都跳过 pair update | 无地址通信的训练基线 |
| Adjacent | 每个阶段都重复 bit-0 相邻配对 | 控制额外深度和参数 |
| Butterfly | 阶段 `s` 使用 `i XOR 2^s` | changing-bit 稀疏通信 |

三个实验臂具有完全相同的已分配参数数量 `34,445,832`。通信 kernel 的输出层从零初始化，因此三个臂从相同函数起点出发。样本、初始化、batch 顺序、优化器、训练轮数、Decoder 与测试集全部匹配。需要注意：Identity 虽然分配了通信参数，却从不执行通信，所以“参数数量相同”不等于“有效计算路径完全相同”；Butterfly 与 Adjacent 才是更接近的等计算量拓扑对照。

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

训练运行于单张 RTX 3090。生成指标是项目代码中固定实现的 token BLEU-4。

### 7.4 预注册判定

Butterfly 必须在至少两颗种子中同时满足：

1. 测试 NLL 优于 Identity 至少 `0.02`；
2. 优于 Adjacent 至少 `0.015`；
3. 长源子集收益不消失；
4. 训练后关闭通信至少恶化 `0.02` NLL，作为依赖检查而非拓扑优势证明；
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

三个实验种子全部通过当时的预注册 gate。严格按当前修订后的证据口径，该表支持：在这个数据、规模和优化合同下，从头训练的 Butterfly 配置稳定优于 Identity 与 Adjacent 配置。它尚未证明任意任务上的普遍优势，也没有仅靠一项运行时干预隔离出 changing-bit 拓扑的全部因果贡献。

### 8.2 长源与运行时依赖检查

在 25--32-piece 长源子集上，Butterfly 相对 Identity 的 NLL 收益分别为：

```text
0.07451 / 0.07825 / 0.10424
mean = 0.08567
```

训练完成后还进行了运行时结构替换：

| 干预 | 平均结果 |
|---|---:|
| Butterfly 改为 Identity | NLL 恶化 1.16873 |
| Butterfly 改为 Adjacent | NLL 约恶化 1.06183 |
| Source shuffle | NLL 约恶化 3.15625 |

这些数字表明已训练模型对原生内部协议敏感，但三种干预的解释力度不同：

- `Butterfly -> Identity` 完全绕过 `B_theta`。后续 FOLD 与 Decoder 接收到未经训练时变换的坐标，因此性能下降几乎是预期现象。它只证明模型依赖 `B_theta`，不能证明 Butterfly 地址拓扑本身更优。
- `Butterfly -> Adjacent` 保留相同通信 kernel 和调用深度，只改变配对调度，因而更接近拓扑干预；但它仍把训练时协议替换为训练外协议，包含分布偏移。
- `Source shuffle` 证明输出依赖源输入，不直接判断 Butterfly 调度。

因此，本节把过去的“因果消融”改称“运行时依赖检查”。Butterfly 的主要正向证据来自 8.1 节的三种子匹配训练比较，而严格的同 checkpoint 拓扑因果实验仍待完成。

### 8.3 严格拓扑消融应当怎样做

下一轮实验必须让 `B_theta` 始终参与，并保持 kernel 参数、调用次数、阶段数、输入、checkpoint 和评估样本完全相同，只改变地址配对图。例如，对宽度 8 的状态：

```text
Native Butterfly:
stage 0: (0,1) (2,3) (4,5) (6,7)
stage 1: (0,2) (1,3) (4,6) (5,7)
stage 2: (0,4) (1,5) (2,6) (3,7)

Repeated adjacent:
stage 0..2: 始终重复 (0,1) (2,3) (4,5) (6,7)

Random perfect matching:
每阶段使用固定种子的随机两两配对，保持相同边数与 kernel 调用次数
```

对同一批句子计算成对差值：

$$
\Delta L_i = L_i(\text{wrong topology}) - L_i(\text{native topology}).
$$

运行时配对实验若稳定恶化，只能说明模型依赖训练时地址图。为了进一步判断 changing-bit 调度是否提供了更好的学习偏置，还必须把 Native、Repeated Adjacent 和若干具有相同边数、调用次数及全局覆盖能力的固定 Random Matching 从头训练。只有“同 checkpoint 拓扑干预”和“多种子匹配重训”同时支持，才把“收益来自 changing-bit 拓扑”升级为严格支持。

### 8.4 数学完整性

三颗种子中：

```text
Butterfly forward/inverse MSE：约 6.7e-16 .. 7.2e-16
FOLD/UNFOLD closure MSE：约 6.7e-15 .. 8.2e-15
dense N x N attention allocation：无
```

可逆误差接近 FP32 数值精度，说明任务训练没有破坏代数闭合。

### 8.5 改进链及其效果

最终结果之前经历了三次决定性的改进。由于各阶段的数据规模和实验合同不同，下表不能被当作一条严格的单变量曲线；它用于说明每次修改解决了什么问题，以及留下了什么可复查结果。

| 改进 | 实验规模 | 主要结果 | 当时得到的结论 |
|---|---:|---|---|
| 完整 lifting state：root + details + recursive READ | 27K/2K/2K，10 epochs | recursive/root/full NLL 为 `5.0903/5.4337/5.1342`；source/root shuffle 损伤 `+1.4450/+1.7204`；闭包 MSE `1.73e-14` | Decoder 确实同时使用 root 和多个 detail 深度；root 单独不够 |
| 可学习 Update：`0.5d + 0.5 tanh(U_theta(d))` | 200K/5K/5K，5 epochs | 固定到可学习 Update：NLL `4.6743 -> 4.6335`，token BLEU-4 `9.609 -> 9.909`；闭包 MSE `2.35e-14` | 可以在不破坏 UNFOLD 的情况下，让梯度调整 detail 向 parent 的贡献 |
| XOR-Butterfly 通信 | 200K/5K/5K，3 seeds | 平均 NLL `4.56509`，BLEU-4 `10.5462`；优于从头训练的 Identity 与 Adjacent | 在该训练合同下，Butterfly 配置有稳定收益；严格拓扑因果性仍待补证 |

这三步对应一条清晰的算法演化：先证明完整 `H` 可读，再让抽水规则可学习，最后补上 leaf 平面的长距离通信。每一步都保留前一步的闭包要求，而不是用新的黑盒覆盖旧问题。

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

复核代码后发现，这里的 native NLL 使用全部验证行，而 runtime Identity 仅使用前 512 行，两者不是同一个评估样本集合。因此 `4.80849` 与 `+1.32457` 只能作为历史诊断记录，不能进入正式 Claim，也不能称为配对消融结果。中期 native NLL 仍可描述规模训练自身的数值进展。修复方法是预先冻结同一批 probe rows，再分别运行 Native、Adjacent 和固定随机配对，并报告逐句成对差值。

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

### 10.4 当前代价与研究边界

TreeHeap 的显式层级与可逆 detail 允许研究者直接做地址替换、深度开放、root/detail shuffle 和通信调度干预。这是当前最明确的可观察性收益。代价也同样明确：当前递归 Decoder 会展开总计小于 `2N` 个层级节点，并在每个输出时间步进行软 READ，生成复杂度约为 `O(TN)`；当前 checkpoint 也没有实现文件大小意义上的压缩。

因此，“多分辨率”在本文中指状态被组织成 root 与逐层 detail，而不是已经节省了存储。“可逆”指在数值精度内能由 `H` 恢复通信前状态，而不是保证生成文本无损。“私有协议”指 Encoder 与 Decoder 从任务梯度共同形成内部编码，而不是已经发现了人类可读的语义坐标系。

## 11. 局限与风险

1. **翻译质量仍处于研究 PoC。** 实验一 BLEU-4 约 10.55，不能作为商业翻译器使用。
2. **全量训练尚未完成。** 实验二所有数字均为中期快照。
3. **Decoder 计算仍然昂贵。** 当前实现展开总计小于 `2N` 个层级节点，每个输出步进行软 READ，生成大致为 `O(TN)`。
4. **地址语义未知。** XOR 地址是通信调度，不等于已经发现语义坐标系。
5. **数据存在噪声。** WMT massive 含网页、商品、成人语句和实体错配；模型输出会继承这些问题。
6. **评估尚不完整。** 最终稿需要补充标准 BLEU、chrF、COMET、重复率、长度分桶和人工盲评。
7. **硬件范围有限。** 当前主要证据来自单张 RTX 3090；跨硬件和多卡可扩展性尚未验证。
8. **严格拓扑消融尚未完成。** 运行时 Identity 绕过了整个已学习通信变换；全量训练脚本还存在 native/Identity 评估样本未配对问题。它们不能替代保持 `B_theta` 活跃、仅改变配对图的严格实验。

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

TreeHeap 从一个关于有序树、旋转切片和有限内存的直觉出发，经过了一条并不笔直的研究路径。路径地址曾被误当成语义，随机张量的可区分性曾被误写成结构能量，flat 路由曾伪装成递归路由，几何 feature 曾直接泄漏答案，root 也曾被要求独自承担整个句子。每一次失败都迫使算法增加更严格的定义：地址与语义分离，`theta` 与 `H(x)` 分离，root 与完整状态分离，数学闭包与归纳学习分离。

当前算法由四个相互衔接的部分组成：token embedding 把输入写入叶地址；共享的 XOR-Butterfly 双节点 kernel 在固定容量内建立 changing-bit 通信；可逆 lifting 把叶状态分解为 root 和多层 detail；递归概率 READ 在生成时按时间步读取不同分辨率。最终 token 交叉熵沿着整条链反向传播，使 Encoder 与 Decoder 在没有人工内部标签的情况下共同形成私有协议。

三种子匹配训练比较说明，在当前实验合同下，Butterfly 配置得到的 NLL 与 BLEU-4 优于 Identity 和 Adjacent 配置。运行时旁路只说明已训练模型依赖 `B_theta`，不能单独证明 changing-bit 拓扑优势；这个边界现已明确写入证据链。更早的 lifting 实验说明 root 与多个 detail 深度都具有因果作用，可学习 Update 又在保持数值闭包时改善了任务结果。全量双向实验进一步显示，固定探针可以从重复坍缩逐渐长出保留年份、事件和方向的翻译轮廓。

TreeHeap 仍不是完成品。它的自由生成仍有重复、实体偏差和不自然表达，当前 READ 代价较高，内部地址也还不能被直接解释。但它已经不再只是一组散落的哲学比喻或 toy 公式，而是一套有代码、有 checkpoint、有闭包测试、有匹配对照、有依赖检查、有失败档案并能继续接受数据训练的算法。本文所记录的，正是这套算法怎样被两位研究参与者一问一答地构造出来，并第一次开始形成语言协议。

## 数学背景文献

1. Sweldens, W. *The Lifting Scheme: A Custom-Design Construction of Biorthogonal Wavelets*. Applied and Computational Harmonic Analysis, 1996.
2. Dao, T., Gu, A., Eichhorn, M., Rudra, A., and Ré, C. *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*. ICML, 2019. [arXiv:1903.05895](https://arxiv.org/abs/1903.05895).
3. Ebrahimi-Fard, K., and Rahm, L. *A Survey on the Munthe-Kaas-Wright Hopf Algebra*. 2023. [arXiv:2306.04381](https://arxiv.org/abs/2306.04381).

## 附录 A：Claim 边界表

| 陈述 | 当前状态 |
|---|---|
| TreeHeap FOLD/UNFOLD 在训练后保持数值可逆 | 支持 |
| XOR-Butterfly 在固定容量中提供全地址稀疏路径 | 数学定义成立 |
| Butterfly 训练配置在当前 WMT 合同中优于 Identity/Adjacent | 三种子支持 |
| 已训练模型依赖通信变换 `B_theta` | 运行时旁路支持，但属于弱依赖证据 |
| 收益严格来自 changing-bit 拓扑而非协议替换或分布偏移 | 开放，待严格配对拓扑消融 |
| 一个共享 TreeHeap 能开始形成中英双向协议 | smoke 与中期结果支持 |
| root 是人类可读摘要 | 未证明 |
| XOR 地址天然具有语义 | 未证明 |
| 当前模型达到产品级翻译质量 | 否 |
| 当前实现节省训练或生成计算 | 未证明 |

## 附录 B：最终稿待补实验

1. 完成 96 小时双向训练并冻结最终 checkpoint；
2. 在固定测试集报告标准 BLEU、chrF、COMET；
3. 报告 8--32、33--64、65--128、129--253 长度区间；
4. 报告自由生成重复率、空输出率与 unique-output rate；
5. 对最终 checkpoint 使用同一批样本重做 Native、Adjacent、固定随机完美匹配与 source shuffle；Identity 仅保留为模块依赖检查；
6. 对拓扑干预报告逐句配对 NLL 差值、bootstrap 置信区间和多种固定随机配对，并对相同拓扑重新做多种子匹配训练；
7. 对 `dreams` 进行时间序列分析，但不把探针选择成训练目标；
8. 增加独立人工盲评与数据污染审计；
9. 测量实际 wall-clock、GPU hours、token throughput 与显存。
