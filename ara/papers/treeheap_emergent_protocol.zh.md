# TreeHeap：可逆多分辨率树状态、稀疏通信与双语序列协议

**English title:** *TreeHeap: Reversible Multiresolution Tree States, Sparse Communication, and a Bilingual Sequence Protocol*

**状态：** 中文审阅稿 v0.5，2026-08-02

**作者：** Houming818（Independent Researcher）

**研究设计：** Houming818 与 Codex（Review Engineer）协作完成

**审计协作：** DeepSeek、GLM

**代码与证据：** SameTime / ARA，GPL-3.0

**证据截止：** 正式结论截至三种子 WMT 匹配实验；全量双向训练仍在运行，其数字和输出仅作为阶段观察。

## 摘要

本文提出 TreeHeap：一种固定容量、显式寻址、可逆且具有多分辨率状态的序列模型。它研究的问题不是“能否把数组换成树保存”，而是：当模型状态本身具有 root、内部节点、leaf、路径和子堆时，最终任务梯度能否让 Encoder 与 Decoder 形成一套共同使用的内部协议。

TreeHeap 的最终架构由四个部分组成。首先，输入 token 被写入叶地址。其次，共享的双节点可逆 kernel 按 XOR-Butterfly 调度进行稀疏通信，使固定容量为 (N) 的叶状态在 (log_2N) 个阶段内获得全地址通信路径。随后，可逆 lifting FOLD 把叶状态分解为一个 root 与逐层 detail；它改变表示分辨率，但保留精确 UNFOLD 所需的信息。最后，递归概率 READ 根据 Decoder 隐状态，在 root、内部节点与 leaf 之间分配读取质量。系统没有接收人工语法树、主谓宾标签、角色槽位或内部节点监督，训练目标只有目标序列的 token 交叉熵。

在真实 WMT 中英语料上的参数匹配实验中，我们保持参数量、初始化、样本、批次顺序、优化器与 Decoder 一致，只改变通信调度。三颗随机种子的平均测试负对数似然（NLL）分别为：关闭通信 (4.65196)、重复相邻通信 (4.65415)、XOR-Butterfly 通信 (4.56509)；对应项目内 token BLEU-4 为 (9.9501)、(9.9485) 与 (10.5462)。该结果支持一个有限结论：在当前数据、规模和训练合同中，从头训练的 Butterfly 配置稳定优于两个注册对照。训练后旁路通信会显著损害 NLL，但这一干预同时移除了已经学习的坐标变换，因此只能说明模型依赖该模块，不能单独证明 changing-bit 拓扑的全部因果优势。

我们还启动了一个使用 1417 万中英平行句对、双向交替、最长 253 pieces 的 96 小时单卡实验。截至本文修订时，任务已处理约 3692 万训练样本和 7.81 亿目标 token。验证 mean NLL 的最好阶段观测为 (3.4230)，最近一次为 (3.4302)，表明后期训练进入震荡平台，而非继续单调下降。固定 `dreams` 探针从初始的单 token 重复，逐渐发展为能够保留年份、事件和部分关系的源相关句子，但输出仍存在重复、实体偏差和不自然表达。因此，这部分只构成协议生长的观察记录，不构成最终质量结论。

TreeHeap 当前最可靠的结论是：可逆树形状态、稀疏地址通信、递归多分辨率读取和最终序列损失能够组成一条可训练的端到端路径。它已经成为一个可以复现、比较、干预、恢复和继续训练的算法对象；语义地址、存储压缩、计算优势和产品级生成仍然是开放问题。

## 1. 引言

### 1.1 从“模型参数”到“模型状态的形状”

机器学习通常把长期知识保存在共享参数中，并为每个输入计算临时隐状态。TreeHeap 接受这一基本事实，但进一步追问：如果临时状态不是一排彼此平等的向量，而是一棵有地址、有父子关系、有递归深度的树，学习过程会发生什么？

一棵树天然提供四类序列之外的对象：

1. **地址：** 一个状态位于哪个节点；
2. **路径：** 从 root 到该节点经过哪些分支；
3. **子结构：** 一个内部节点覆盖哪些后代；
4. **分辨率：** leaf 保存局部细节，较高节点覆盖更大的范围。

这些性质本身不等于语义。把 token 放在树上，不会自动产生语言理解；把两个节点连起来，也不会自动得到句法关系。TreeHeap 的核心问题恰恰是：怎样让任务数据通过梯度选择和使用这些结构，而不是由研究者事先替模型写入答案。

### 1.2 研究问题

本文围绕三个可以证伪的问题展开。

**问题一：代数可行性。** TreeHeap 的通信、FOLD 与 UNFOLD 是否有明确数学定义，并在训练后保持数值可逆？

**问题二：结构参与性。** root、detail、深度和叶地址通信是否真正影响任务结果，还是模型最终退化为换一种形状保存数组？

**问题三：协议形成。** 在没有内部语义标签的条件下，Encoder 与 Decoder 能否只依赖目标 token loss，形成一套共享、可训练、可恢复的内部编码协议？

本文将“私有协议”定义为由任务训练产生、由 Encoder 与 Decoder 共同使用、但没有被人工命名的内部表示规则。本文将“涌现”限定为一个操作性概念：能力由最终任务目标间接形成，而不是由内部标签直接指定。它不表示神秘过程，也不等同于意识、世界模型或人类可读语义。

### 1.3 本文贡献

本文的主要贡献如下。

1. 定义固定容量 TreeHeap 状态，严格区分共享参数 (	heta) 与样本状态 (H_	heta(x))。
2. 构造可逆 lifting FOLD/UNFOLD，使完整状态由 root、逐层 detail 与 mask 共同组成。
3. 引入由共享双节点 kernel 组成的 XOR-Butterfly 通信，在不分配稠密 (N\times N) 注意力矩阵的条件下提供全地址通信路径。
4. 定义递归概率 READ，让 Decoder 在每个生成时间步选择读取分辨率，而不是预先规定 root 或 leaf 必须承担全部信息。
5. 设计三种子、参数匹配、数据匹配的真实 WMT 比较，区分“从头训练的架构差异”与“训练后运行时依赖”。
6. 建立可恢复训练、固定 `dreams` 探针和 ARA Claim/Evidence 记录，使失败实现、降级结论和中间生成均可复查。

本文不声称 TreeHeap 已达到产品级翻译质量，也不声称它已经节省存储或计算。当前工作的目标是建立一种新的、可训练且可检验的状态结构，而不是宣布一个已经完成的通用模型。

## 2. 从树形直觉到可训练算法

TreeHeap 不是从一张完整架构图开始的。它由 Houming818 提出的结构直觉、Codex 的数学与代码转写，以及 DeepSeek、GLM 的交叉审计反复修订而来。本节保留这段演化过程，因为最终算法中的每个主要部件，都对应一个被实验暴露出来的具体缺口。

### 2.1 路径能够定位，但不能自动解释语义

早期 TreeHeap 使用由 token ID 决定的路径。路径拥有公共前缀、左右分支和相邻地址，于是最初的设想是：公共前缀更长的 token 也许属于同一语言结构，地址更近的 token 也许应获得更高连接权重。

这个设想混淆了“位置”与“含义”。同一个 token 在不同句子中可以承担不同作用，但由 token ID 产生的基础路径不会随上下文改变。两个词表 ID 接近的 token 也不必在当前句子中相关。路径能够可靠回答：

```text
这个状态被放在哪里？
```

却不能单独回答：

```text
这个状态在当前上下文中表示什么？
```

因此，最终设计保留地址，却不把地址直接命名为语义。

### 2.2 位权与张量解决“可区分”，没有解决“可选择”

Houming818 曾用数字 (321) 与 (123) 解释结构位权：数字不同，不仅因为包含的基元不同，也因为基元处于百位、十位和个位。对应到语言，一个 token 向量 (s_i) 可以与角色基 (e_r) 做外积：

$$
T=\sum_i s_i\otimes e_{r_i}.
$$

这样，同一个 token 放在不同角色基上会得到不同表示。实验也确认，拼接、外积与非交换组合能够区分不同排列。

但这只证明了表示容量。若在构造张量前已经知道哪个 token 是 SUBJECT、OBJECT 或 ROOT，那么结构答案已经由外部标签提供。随机角色基能够让两个排列不同，也不能保证正确排列自然获得最低能量。于是研究问题从“能否表示两个结构”升级为：

> 最终任务梯度能否让正确结构在候选表示中获得稳定优势？

这一转折使项目逐步离开人工语法角色，转向由数据形成的 latent protocol。

### 2.3 概率容器能够延迟决定，但不能创造信息

早期 Graph Builder 还尝试保留多个 parent 候选，而不是立即选择唯一父节点。例如：

```text
Parent A: 0.62
Parent B: 0.25
Parent C: 0.13
```

该设计体现了一个后来被继承的原则：信息不足时保留概率分布，不要过早坍缩。历史 parent 候选实验曾获得很高的 top-k gold 覆盖率，说明候选桶可以避免过早丢失答案。

然而，概率容器只能保存已有候选。若节点状态没有携带可区分信息，延迟选择不会自动产生结构。最终 TreeHeap 保留了概率 READ 的思想，但把概率对象从“人工 parent 候选”改成了对真实 root、internal state 和 leaf 的递归读取质量。

### 2.4 世界模型参考系没有由向量距离自动产生

项目曾把上下文背景场称为“世界模型”，希望得到类似下列可迁移关系：

```text
ball + foot -> football
ball + hand -> basketball
```

围绕 `t_merge`、CMul、去中心化和 relation anchor 的实验说明，旧 checkpoint 中存在强公共方向；部分差异在去中心化或 merge 前仍然存在，但没有形成稳定、跨样本迁移的关系方向。历史 TreeHeap checkpoint 的向量平均 cosine 一度达到约 (0.985)，表现出严重方向坍缩。

因此，本文不把“向量之间存在距离”称为世界模型。世界参考系至少需要在新样本上重复出现关系方向，并稳定优于困难负例。这个 Claim 在当前证据中仍未成立。

### 2.5 从封闭算子到可逆信息抽取

第一次大退航之后，研究回到 M0 数学层。TreeHeap 被要求拥有明确的 `Zero`、`plus`、`diff`、compose、decompose、mirror 和子堆 kernel。这里逐渐形成了两类证明纪律：

- 由定义直接成立的演绎性质，用闭包、逆运算和数值误差验证；
- 必须从数据得到的归纳性质，用 loss、梯度、对照和干预验证。

最简单的 parent 是对子节点求和或平均。这种操作能减少节点数，却丢失左右关系和子堆身份。紧凑 route 实验表明，随机 token 向量求和后，Decoder 无法稳定恢复目标。Houming818 用“信息抽水机”描述希望看到的过程：leaf 保存具体状态，parent 接收更大范围的信息，越靠近 root 分辨率越粗；但被抽走的细节不能凭空消失。

Lifting scheme 提供了所需的数学形式：parent 保存更新后的 anchor，detail 保存预测残差。FOLD 改变分辨率，UNFOLD 又能恢复完整状态。由此，TreeHeap 不再要求 root 单独记住整句。

### 2.6 root 不是完整 TreeHeap 状态

早期 Decoder 多次选择最短优化路径，只读取 root。root-exclusive 实验确实增强了 root 的因果性，却损害了翻译 NLL。这说明“信息向上流动”不等于“所有信息必须挤进 root”。

最终定义改为：

$$
H=\text{root}+\text{all details}+\text{masks}.
$$

root 是完整状态的一部分，不是完整状态本身。Decoder 在每个生成时间步决定在哪个深度停止、向哪个 child 继续。本文也不预设“越靠近 root 一定越像人类摘要”；不同深度具体承载什么，只能由任务训练与干预实验确定。

### 2.7 局部 FOLD 无法直接解决长距离 leaf 通信

二叉 FOLD 的局部性带来另一个问题：相距很远的 leaf 要经过多层 parent 才能相遇，途中还会受到分辨率变换影响。循环位移、扩大邻接窗口和矩阵视觉层都曾被讨论，但要么覆盖不完整，要么缺少可逆性和固定容量约束。

最终采用 XOR-Butterfly，因为它同时满足：

1. 不增加 TreeHeap 容量；
2. 只使用共享局部双节点 kernel；
3. 在 (log_2N) 个阶段内提供所有叶地址之间的通信路径；
4. 每个阶段都有明确逆运算。

它把“旋转、换观察方向”的几何直觉收敛成一个有限、可计算、可审计的地址调度。

### 2.8 失败如何约束最终算法

| 早期实现或假设 | 暴露的问题 | 进入最终设计的修正 |
|---|---|---|
| token ID 路径直接解释语义 | 地址不随上下文角色改变 | 地址与语义学习分离 |
| 随机角色基张量能量 | 可区分不等于可选择 | 结构选择交给任务梯度 |
| 概率 parent 桶 | 能延迟丢失，不能创造状态信息 | 概率用于真实递归 READ |
| 旧 TreeHeap checkpoint | 向量方向高度坍缩 | 撤回背书，重新训练 |
| 每种长度一张 flat route 表 | 记住长度，没有共享递归规律 | 使用共享 kernel |
| route feature 包含左右区间答案 | 特征泄漏产生虚假高准确率 | kernel 只读 query 与实际状态 |
| parent 等于子节点加和 | 次序和子堆身份丢失 | 保存 lifting detail |
| root-exclusive Decoder | root 因果性增强但任务质量下降 | 完整 (H) 参与 READ |
| 固定 lifting Update | 数学稳定但任务适应性有限 | 增加可学习 Update |
| 仅局部相邻 FOLD | 长距离地址通信过深 | FOLD 前加入 Butterfly |
| 只观察平均 NLL | 重复坍缩可能被均值掩盖 | 固定 `dreams` 与重复率审计 |

这张表不是研究花絮，而是最终算法的约束来源。TreeHeap 的每一项核心设计，都必须对应一个已经被观察到的失败模式。

## 3. TreeHeap 的形式化定义

### 3.1 地址、容量与有效位置

给定最大叶容量 (N=2^D)，TreeHeap 使用二叉堆地址：

$$
\operatorname{root}=0,
\qquad
\operatorname{left}(i)=2i+1,
\qquad
\operatorname{right}(i)=2i+2.
$$

输入 token 首先形成叶状态：

$$
X^{(0)}=[x_0,x_1,\ldots,x_{N-1}],
\qquad x_i\in\mathbb{R}^{m}.
$$

实际序列不足 (N) 时，剩余位置由 mask 关闭。规模化版本的最大叶容量为 256；较短 batch 只展开到能够容纳自身的最近二次幂。不同宽度共享相同 kernel 参数，不是多个独立模型。

### 3.2 参数 \(\theta\) 与样本状态 \(H_\theta(x)\)

TreeHeap 中最容易混淆的是长期参数和临时状态。

共享参数记为：

$$
\theta=\{E_{src},E_{tgt},F,G,\alpha,P,U,E_{depth},S,B,
\operatorname{GRU},W_o\}.
$$

其中：

- (E_{src}) 与 (E_{tgt}) 是输入、输出 embedding；
- (F,G,\alpha) 定义 Butterfly 双节点 kernel；
- (P,U) 是 lifting Predictor 与 Update；
- (E_{depth},S,B) 定义深度表示、stop 和 branch；
- GRU 与 (W_o) 根据读取上下文生成目标 token。

对于具体输入 (x)，这些共享参数计算临时状态：

$$
H_\theta(x)
=\left(r_x,\{d_x^{(0)},d_x^{(1)},\ldots,d_x^{(D-1)}\},M_x\right).
$$

(H_	heta(x)) 随输入改变，不作为一份新的模型参数永久写入 checkpoint。可以把 (	heta) 理解为长期形成的编码规则，把 (H_	heta(x)) 理解为规则对当前句子的实例化。

### 3.3 WRITE：把 token 写入 leaf

输入由方向 token、SentencePiece pieces 与 EOS 组成。方向不是额外旁路变量，而是序列的第一个特殊 token。对有效位置：

$$
x_i=E_{src}(w_i).
$$

其中 (w_0) 是 `en2zh` 或 `zh2en` 方向 token，后续位置是原文 pieces 与 EOS。当前实现没有额外的位置 embedding；位置差异由 leaf 下标、Butterfly 配对和二叉 FOLD 路径进入计算。无效位置由 mask 关闭，避免 padding 在通信和 FOLD 中被误当成内容。

WRITE 本身没有产生高层结构。它只把离散输入映射到共享连续空间，并放入可寻址位置。

### 3.4 可逆双节点通信 kernel

对于一对状态 ((a,b))，第 (s) 个通信阶段定义：

$$
b'=b+\alpha_s\tanh(F_\theta(a)),
$$

$$
a'=a+\alpha_s\tanh(G_\theta(b')).
$$

(F_	heta) 与 (G_	heta) 是所有地址共享的小型非线性 kernel，(alpha_s) 是有界阶段增益。逆运算按相反顺序进行：

$$
a=a'-\alpha_s\tanh(G_\theta(b')),
$$

$$
b=b'-\alpha_s\tanh(F_\theta(a)).
$$

因此，通信改变状态坐标，却不要求在数学上丢弃输入。

### 3.5 XOR-Butterfly 地址调度

在阶段 (s)，地址 (i) 与地址

$$
j=i\oplus2^s
$$

配对，其中 (oplus) 是按位 XOR。每阶段包含 (N/2) 个双节点操作，共有 (log_2N) 个阶段，总 pair 操作数为：

$$
\frac{N}{2}\log_2N=O(N\log N).
$$

对 (N=8)，配对图为：

```text
stage 0: (0,1) (2,3) (4,5) (6,7)
stage 1: (0,2) (1,3) (4,6) (5,7)
stage 2: (0,4) (1,5) (2,6) (3,7)
```

经过全部阶段，每个地址都存在到其他地址的 changing-bit 路径。XOR 是明确加入的通信先验，不是已经发现的语义坐标。任务梯度只能决定如何使用这些路径，不能反过来证明路径天然具有语言含义。

### 3.6 可逆多分辨率 FOLD

对相邻左右状态 ((l,r))，lifting 定义：

$$
d=r-P_\theta(l),
$$

$$
p=l+U_\theta(d).
$$

(p) 进入上一层成为 parent，(d) 作为本层 detail 保存。逆运算为：

$$
l=p-U_\theta(d),
$$

$$
r=d+P_\theta(l).
$$

递归执行后，(N) 个叶状态被组织为一个 root 与总计 (N-1) 个逐层 detail。当前 Update 使用稳定线性项与可学习项的组合：

$$
U_\theta(d)=0.5d+0.5\tanh(\widetilde U_\theta(d)).
$$

可学习项从零初始化，所以训练开始时系统等价于确定的 (0.5d) 更新；随后梯度可以调整 detail 对 parent 的贡献，同时保持逆式成立。

### 3.7 递归概率 READ

Decoder 在输出时间步 (t) 维护隐状态 (h_t)。从 root 开始，每个候选节点 (n_i^{(k)}) 计算停止概率：

$$
p_{stop}(i,k,t)
=\sigma\left(S_\theta\left[q(h_t),n_i^{(k)}+e_k\right]\right).
$$

没有停止的概率质量继续分配给左右 child：

$$
p(c\mid i,t)
=\operatorname{softmax}_c
\left(\frac{B_\theta(h_t)^\top n_c}{\sqrt m}\right).
$$

如果到达节点 (i) 的质量为 (m_i)，则：

$$
m_i^{stop}=m_i p_{stop},
$$

$$
m_{left}+m_{right}=m_i(1-p_{stop}).
$$

于是每层递归都满足质量守恒：

$$
m_i^{stop}+m_{left}+m_{right}=m_i.
$$

所有深度上停止节点的加权和形成上下文 (c_t)。Decoder 随后更新：

$$
h_{t+1}=\operatorname{GRU}([E_{tgt}(y_t),c_t],h_t),
$$

$$
p(y_{t+1})=\operatorname{softmax}(W_o[h_{t+1},c_t]).
$$

### 3.8 训练目标与梯度路径

训练使用 teacher forcing，唯一语言目标是目标 token 的交叉熵：

$$
\mathcal L_{seq}
=-\sum_t\log p_\theta(y_t\mid y_{<t},H_\theta(x)).
$$

梯度依次穿过输出层、GRU、递归 READ、UNFOLD、root/detail、lifting、Butterfly 和输入 embedding。没有额外损失告诉某个内部节点应当表示“主语”“食物”或“时间”。若 Encoder 与 Decoder 形成了共同内部协议，它只能来自这条端到端梯度链。

完整前向过程可以概括为：

```text
input token
    -> WRITE
    -> reversible Butterfly communication
    -> reversible multiresolution FOLD
    -> H(root, details, masks)
    -> recursive probabilistic READ
    -> recurrent generation state
    -> target-token distribution
    -> cross-entropy loss
```

## 4. 私有协议与涌现的可检验定义

### 4.1 私有不等于不可验证

本文所说的“私有协议”不要求人类为每个内部维度命名。它要求以下可观察条件同时成立：

1. 内部节点没有人工语义标签；
2. Encoder 与 Decoder 由同一最终任务共同训练；
3. 相同参数能够处理不同输入和长度；
4. 改变输入会改变状态和输出；
5. 干预协议结构会造成可测量损失；
6. checkpoint 保存、重载和恢复训练后行为可复现。

协议可以是私有的，但协议是否存在必须接受公开实验。仅仅看到向量、树或非零梯度，都不足以证明协议已经形成。

### 4.2 涌现不等于无因解释

这里的“涌现”有三个边界。

第一，内部结构不是由句法标签直接监督，而由最终序列目标间接形成。第二，模型行为可以随着数据量出现阶段变化，例如由高频重复转向源相关输出。第三，变化必须能够由固定输入、固定 checkpoint 和固定评估程序复查。

“涌现”不表示结构形成原理不可讨论。WRITE、Butterfly、FOLD、READ 和梯度路径都是公开设计；私有的是训练后形成的具体坐标与分工，而不是形成协议的物理通道。

### 4.3 证据分级

本文把证据分为四级。

| 等级 | 回答的问题 | 典型证据 |
|---|---|---|
| E1 代数正确性 | 运算是否按定义成立 | inverse/closure MSE |
| E2 机制参与性 | 模块是否被任务使用 | root/detail/source 干预、深度读取质量 |
| E3 架构比较 | 某配置是否在匹配合同中更优 | 多种子从头训练对照 |
| E4 规模生长观察 | 行为是否随训练数据发生变化 | 验证曲线、固定 `dreams` |

E1 不能推出语言能力；E2 不能推出架构优势；E3 不能推出跨任务普遍性；E4 的主观样例不能代替标准测试指标。后文按这一等级组织实验。

## 5. 实验设计

### 5.1 实验一：代数闭合与递归读取

早期 lifting-pump 实验首先验证完整 (H) 是否可读。实验比较：

- root-only：只允许 Decoder 读取 root；
- full：直接读取完整 UNFOLD 状态；
- recursive：通过 stop/left/right 递归分配读取质量；
- flat：相同任务下的序列读出对照。

同时对 source、root、每一层 detail 与 pairing 进行干预，检查任务损失是否上升。

### 5.2 实验二：Butterfly 匹配 WMT 比较

本实验比较三个从头训练的实验臂。

| 实验臂 | 通信调度 | 目的 |
|---|---|---|
| Identity | 分配相同参数，但不执行 pair update | 无通信训练基线 |
| Adjacent | 每个阶段重复相邻配对 | 控制调用深度与参数 |
| Butterfly | 阶段 (s) 使用 (i\oplus2^s) | changing-bit 稀疏通信 |

三个实验臂具有相同的已分配参数量 (34,445,832)。通信 kernel 输出层从零初始化，所以三个实验臂从相同函数起点开始。它们共享样本、初始化规则、batch 顺序、优化器、Decoder 与测试集。

需要明确：Identity 虽然分配通信参数，却不执行通信，所以参数量相同不等于有效计算量完全相同。Adjacent 与 Butterfly 是更接近的等调用次数拓扑对照。

### 5.3 WMT 数据与训练合同

正式三种子实验使用本地 WMT massive 中英 TSV。每颗种子从前 200 万行中确定性抽样：

```text
train / valid / test = 200,000 / 5,000 / 5,000
source / target length = 8..32 SentencePiece pieces
direction = English -> Chinese
epochs = 5
dim / hidden = 256 / 256
seeds = 8104, 8105, 8106
hardware = one RTX 3090
```

主要指标为 NLL；生成指标是项目代码中固定实现的 token BLEU-4。它适合内部匹配比较，但不能直接等同于标准 detokenized BLEU。

### 5.4 预注册判定

Butterfly 在至少两颗种子中需要满足：

1. 测试 NLL 优于 Identity 至少 (0.02)；
2. 优于 Adjacent 至少 (0.015)；
3. 25--32-piece 长源子集收益不消失；
4. source shuffle 至少恶化 (0.50) NLL；
5. 多个 TreeHeap 分辨率被 READ 使用；
6. 梯度和生成有限且非空。

训练后旁路通信至少恶化 (0.02) NLL，被预注册为模块依赖检查，不被单独解释成拓扑优势。

### 5.5 实验三：全量双向生长观察

规模实验使用：

```text
parallel rows = 14,170,275
dataset size = about 2.4 GiB TSV
directions = en2zh and zh2en
maximum content = 253 pieces
maximum TreeHeap leaves = 256
stream block = 50,000 rows
time budget = 96 hours
```

方向由显式 token 指示，Encoder、Butterfly、FOLD/UNFOLD、READ 与 Decoder 参数全部共享。训练在 block 边界原子保存 checkpoint，可以在中断后继续。

固定 `dreams.txt` 保存六条不进入训练的中英双向输入。每约 100 万训练样本，当前 checkpoint 对这些输入自由解码并保存不可变快照。Dreams 不参与 loss，只用于观察行为生长和坍缩。

## 6. 结果一：代数闭合与多分辨率读取

### 6.1 数值可逆性

在正式 Butterfly 三种子实验中：

```text
Butterfly forward/inverse MSE: about 6.7e-16 .. 7.2e-16
FOLD/UNFOLD closure MSE:      about 6.7e-15 .. 8.2e-15
```

误差接近 FP32 数值精度。这支持 E1 结论：训练没有破坏所定义的逆运算。它不表示生成文本可以无损恢复，也不表示 root 单独包含全部信息。

### 6.2 root 与 detail 的任务作用

在 27K/2K/2K WMT、10 epochs 的 lifting-pump 实验中：

| 读出方式 | 测试 NLL |
|---|---:|
| recursive READ | 5.0903 |
| root-only | 5.4337 |
| full UNFOLD read | 5.1342 |
| flat sequence | 4.8103 |

source shuffle 与 root shuffle 分别造成约 (+1.4450) 和 (+1.7204) NLL 损伤；所有 detail 深度和 pairing 深度均表现出可测影响。recursive READ 优于 root-only，并接近 full UNFOLD read，说明 Decoder 确实利用多个分辨率，而不是只依赖 root。

该实验同时保留一个负结论：flat sequence 仍优于 TreeHeap (0.2800) NLL。因此，实验支持多分辨率机制参与，不支持翻译质量优势。

### 6.3 可学习 Update

在 200K/5K/5K WMT 实验中，把固定 Update 改为可学习 Update 后：

```text
NLL:          4.6743 -> 4.6335
token BLEU-4: 9.609  -> 9.909
closure MSE:  2.35e-14
```

预注册的 (0.05) NLL 提升门槛没有通过，实际提升为 (0.0408)。因此证据支持较窄的机制结论：任务梯度能够在保持可逆闭合的同时调整 detail 向 parent 的贡献；它不足以支持大幅质量提升。

## 7. 结果二：Butterfly 的三种子 WMT 比较

### 7.1 测试损失

| Seed | Identity NLL | Adjacent NLL | Butterfly NLL | 相对 Identity 收益 | 相对 Adjacent 收益 |
|---:|---:|---:|---:|---:|---:|
| 8104 | 4.62285 | 4.62273 | **4.54546** | 0.07738 | 0.07727 |
| 8105 | 4.66175 | 4.67974 | **4.58915** | 0.07260 | 0.09059 |
| 8106 | 4.67127 | 4.65998 | **4.56066** | 0.11061 | 0.09932 |
| Mean | 4.65196 | 4.65415 | **4.56509** | **0.08687** | **0.08906** |

平均 token BLEU-4 为：

| Identity | Adjacent | Butterfly |
|---:|---:|---:|
| 9.9501 | 9.9485 | **10.5462** |

三颗种子均通过注册门槛。严格结论是：在当前数据、模型规模与训练合同中，从头训练的 Butterfly 配置稳定优于 Identity 和重复 Adjacent 配置。

### 7.2 长源子集

在 25--32-piece 长源子集上，Butterfly 相对 Identity 的 NLL 收益为：

```text
seed 8104: 0.07451
seed 8105: 0.07825
seed 8106: 0.10424
mean:      0.08567
```

收益没有在该实验的较长输入端消失。但训练长度最多只有 32 pieces，这不能替代真正的 64、128 或 253-piece 长程验证。

### 7.3 运行时依赖检查及其边界

训练完成后进行结构替换：

| 干预 | 平均 NLL 变化 |
|---|---:|
| Butterfly 改为 Identity | (+1.16873) |
| Butterfly 改为 Adjacent | 约 (+1.06183) |
| Source shuffle | 约 (+3.15625) |

这些结果说明输出依赖源输入，也依赖训练时的通信变换。但是：

- Butterfly 改为 Identity 会完全绕过 (B_\theta)，后续模块收到训练外坐标；
- Butterfly 改为 Adjacent 保留 kernel 调用，却仍替换为训练外协议；
- source shuffle 只证明输入条件性，不判断通信拓扑。

因此，本文把这些结果称为“运行时依赖检查”，不称为严格拓扑消融。

### 7.4 仍缺少的严格拓扑证据

更严格的实验应在相同 checkpoint、相同验证句、相同 kernel 参数和相同调用次数下，只替换地址配对图，并报告逐句成对差值：

$$
\Delta L_i=L_i(\text{wrong topology})-L_i(\text{native topology}).
$$

此外，还应将 Butterfly、重复相邻调度和若干具有相同边数、阶段数与全局覆盖能力的固定随机调度从头训练。只有同 checkpoint 干预和多种子匹配重训同时支持，才能把“收益来自 changing-bit 拓扑本身”升级为强因果结论。

## 8. 结果三：双向协议的规模生长

### 8.1 NLL 轨迹

全量双向训练仍在运行。下表区分已归档快照与当前远程运行观察。

| 训练样本 | Mean validation NLL | 证据性质 |
|---:|---:|---|
| 0 | 10.5612 | 固定初始快照 |
| 5.99M | 3.5630 | 已归档 dream 快照 |
| 11.98M | 3.4839 | 已归档 dream 快照 |
| 17.93M | 3.4561 | 已归档 dream 快照 |
| 21.93M | 3.4407 | 已归档 dream 快照 |
| 31.88M | **3.4230** | 当前运行中的最好 wake 观察 |
| 36.87M | 3.4302 | 2026-08-02 最近 wake 观察 |

从 0 到约 1200 万样本，验证 NLL 快速下降；之后下降明显放缓，并在 (3.42) 到 (3.47) 附近震荡。最好值没有出现在最新 checkpoint，因此不能把训练描述成持续单调改善。更可能的解释包括：当前容量和优化合同接近平台、流式 block 的数据分布变化、双向任务之间的轻微干扰，或者固定验证集对不同训练阶段敏感。这些解释目前都只是待验证假设。

### 8.2 Dreams 从坍缩到源相关输出

固定探针：

```text
SOURCE: The new company is expected to begin operations in the spring of 2019.
REFERENCE: 新公司预计将在2019年春季开始运营。
```

训练开始时：

```text
意愿 Fe Fe Fe Fe Fe Fe Fe ...
```

约 599 万样本时：

```text
预计于2019年年底,开始新年。
```

约 1198 万样本时：

```text
新年公司预计于2019年,预计在2019年春天开始运营。
```

约 1793 万样本时：

```text
预计于2019年春天开始运营。
```

约 2193 万样本时：

```text
新的公司预计将于2019年春春新公司运营。
```

这条轨迹显示，模型从源无关的单 token 循环，发展为能够保留“公司、2019、春季、开始运营”等主要关系。它也显示生成并非单调改善：1793 万样本时的句子比 2193 万样本时更简洁，而后者重新出现重复和语序问题。

另一条因果句探针：

```text
SOURCE: Why is the window wet? Because the rain was blown against the glass by the wind.
```

从初始的符号重复，发展到包含“雨、湿、风、窗户”等源相关词，但在 2193 万样本时仍输出：

```text
雨湿的湿润湿润湿透,因为风湿透风的风湿是无情。
```

这说明模型已经学习部分词汇和关系条件，却仍会进入局部重复吸引子。NLL 的总体改善并不保证每个自由生成样例都同步改善。

### 8.3 双向协议的不对称

在多数 wake 快照中，zh2en NLL 低于 en2zh。例如 3188 万样本时：

```text
en2zh NLL = 3.7308
zh2en NLL = 3.1153
mean       = 3.4230
```

这说明共享协议并没有让两个方向达到相同难度。差异可能来自 tokenizer、目标语言熵、语料噪声或解码器容量，当前尚未由控制实验分解。后续报告必须分别给出两个方向，不能只报告均值。

### 8.4 为什么 Dreams 不是最终指标

Dreams 的价值是让训练过程可见：它能暴露单 token 坍缩、短语循环、实体错误和源条件变化。但六条人工选择的句子不能代表完整测试分布，也不参与梯度。

正式质量结论仍需要：标准 detokenized BLEU、chrF、COMET、重复率、长度分桶、方向分桶和人工盲评。本文只把 dreams 解释为阶段行为证据。

## 9. 讨论

### 9.1 当前证据真正支持什么

第一，TreeHeap 的核心变换可以在训练后保持数值可逆。第二，Decoder 的确使用 root 与多个 detail 深度，完整状态没有退化为 root-only。第三，在匹配 WMT 合同中，Butterfly 配置跨三颗种子获得更低 NLL 和更高项目内 token BLEU-4。第四，一个共享的 TreeHeap 已经能够在双向数据中从高频坍缩发展出源相关序列。

这些结论共同支持：TreeHeap 不再只是一个数据结构草图。它已经拥有确定的状态、共享参数、梯度路径、训练程序、checkpoint 和可干预行为。

### 9.2 当前证据没有支持什么

本文没有证明：

- root 是人类语言中的摘要；
- XOR 地址天然具有语义；
- TreeHeap 内部形成了可迁移世界模型；
- 当前状态表示具有文件压缩意义上的压缩率；
- 当前实现比稠密模型节省实际 GPU 时间；
- 当前生成达到产品质量；
- 当前结果可以推广到翻译之外的任务。

保持这些边界并不会削弱已有结果。相反，它使下一轮实验可以针对真正未知的部分设计。

### 9.3 TreeHeap 当前最独特的研究价值

TreeHeap 当前最明确的价值不是单一质量数字，而是结构可观察性。研究者可以分别干预：

- 某个 leaf 地址；
- 某条 Butterfly 配对边；
- 某个 FOLD 层级；
- root；
- 某层 detail；
- READ 的停止深度。

这些对象在算法中有明确位置，也有明确逆运算。它们使“模型是否使用结构”可以被拆成多个局部问题，而不是只能观察一个整体隐向量。

### 9.4 多分辨率不是已经实现的压缩

一个 (N)-leaf TreeHeap 在 lifting 后仍保存一个 root 和 (N-1) 个 detail，总状态数量没有减少。当前“多分辨率”表示信息被重组为不同尺度，不表示 checkpoint 或运行内存已经缩小。

真正的压缩需要证明：可以丢弃、量化或延迟加载一部分 detail，同时在给定质量预算下获得更低存储或计算成本。这属于未来的率失真问题，当前没有完成。

### 9.5 复杂度与工程代价

Butterfly 通信需要 (O(N\log N)) 个双节点操作，FOLD/UNFOLD 为 (O(N))。当前递归 Decoder 在每个输出时间步读取总计小于 (2N) 个层级节点，因此生成复杂度约为 (O(TN))。代码尚未把稀疏地址操作优化为高效 CUDA kernel。

所以，数学上的稀疏不等于当前实现已经更快。计算优势必须用训练 token、GPU 小时、显存、吞吐和实际反向 FLOPs 测量，而不能由复杂度公式直接宣布。

## 10. 局限、风险与可证伪条件

### 10.1 当前局限

1. 正式 WMT 比较规模为 20 万训练句、最长 32 pieces，仍属于研究型实验。
2. 全量双向训练尚未结束，运行中最好 checkpoint 不等于最终 checkpoint。
3. 自由生成仍有重复、实体偏差、语法错误和局部吸引子。
4. 当前 Decoder 为 (O(TN))，尚未证明工程效率。
5. WMT massive 含网页、商品、成人内容和错配样本，模型会继承数据噪声。
6. 主要结果来自单张 RTX 3090，跨硬件与多卡扩展性未知。
7. runtime Identity 会绕过已学习变换，不能作为严格 topology-only 消融。
8. 尚无标准 BLEU、chrF、COMET 和人工盲评的完整最终报告。

### 10.2 核心 Claim 的否证条件

以下结果应迫使我们撤回或降级相应 Claim。

| Claim | 否证条件 |
|---|---|
| FOLD/UNFOLD 可逆 | 在有效输入上闭包误差系统性超出浮点误差，且无法由数值精度解释 |
| 多分辨率状态参与任务 | root/detail/depth 干预在多种子上不造成稳定损失，或 Decoder 只依赖 leaf 直通路径 |
| Butterfly 在当前合同中有收益 | 匹配复现实验不能重复三种子收益 |
| changing-bit 拓扑具有特殊优势 | 具有相同边数、阶段数和全局覆盖的固定随机拓扑表现相当或更好 |
| 双向私有协议正在形成 | 输出长期源无关、方向混淆、验证损失不改善或 checkpoint 重载行为不一致 |
| TreeHeap 具有计算优势 | 匹配质量下 GPU 小时、吞吐、显存和 FLOPs 没有改善 |

### 10.3 下一轮最重要的实验

下一轮应优先完成三件事：

1. 冻结全量双向训练的最好 checkpoint，而不是默认采用最后 checkpoint；
2. 在固定验证句上完成 Native、Adjacent 与固定随机 topology 的样本配对评估；
3. 对同覆盖能力的多个拓扑进行匹配重训，并测量质量、吞吐和 GPU 小时。

只有这三步完成，才能判断当前收益主要来自 changing-bit 调度、可逆通信本身，还是额外非线性深度。

## 11. 可复现性

### 11.1 核心代码

```text
ara/s3-generation/src/s2_treeheap_butterfly_wmt.py
ara/s3-generation/src/s3_treeheap_butterfly_bilingual_full.py
ara/s3-generation/src/s2_treeheap_butterfly_cli.py
```

### 11.2 Logic 与 Evidence

```text
ara/s3-generation/logic/treeheap_butterfly_wmt_ablation.md
ara/s3-generation/logic/treeheap_butterfly_bilingual_full_train.md
ara/s3-generation/evidence/s2_treeheap_butterfly_wmt_formal/
ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/
```

正式三种子 checkpoints 约 132 MiB，保留于 `io`，SHA-256 记录在 Evidence 的 `CHECKPOINTS.md`。全量训练由 `taskd` 串行管理，在 50K 行 block 边界原子保存并支持恢复。

### 11.3 关键版本记录

```text
373c0b4  Butterfly WMT preregistration
8f48cbe  formal three-seed evidence
2eef49e  bilingual full trainer
a0feb5b  bilingual smoke evidence
```

### 11.4 ARA 研究流程

本项目使用：

```text
intuition -> Claim -> Predict -> Experiment -> Evidence -> Audit -> Revision
```

Houming818 提出结构直觉并审核算法是否仍然体现 TreeHeap；Codex 负责形式化、实验设计和代码审查；DeepSeek 与 GLM 进行独立复核。路径语义、flat route、geometry feature 泄漏、错误 CLI 标签和运行时消融过强解释，都曾在审计后被公开降级。

ARA 的目标不是保证研究者不犯错，而是保证错误不会在没有记录的情况下变成下一轮前提。

## 12. 相关数学背景

TreeHeap 的最终实现并非从零发明全部数学对象。

Lifting scheme 提供了可逆的 coarse/detail 分解：一个分支用于预测，预测残差作为 detail 保存，另一个分支由 detail 更新。TreeHeap 把这种思想用于可学习的树形序列状态。

Butterfly factorization 研究如何用稀疏分阶段变换表达全局线性变换。TreeHeap 使用 XOR changing-bit 配对建立固定容量的全地址通信，但本文实现的双节点 kernel 是非线性可逆耦合，不等同于某一特定线性 Butterfly 矩阵。

有根树的 Hopf 代数、operad 和 decomposition space 为 compose、cut、grafting 与 many-in-one composition 提供了成熟背景。TreeHeap 与这些理论具有结构对应，但当前数值模型还没有被证明是某个特定 Hopf 代数或 operad 的完整实现。因此，本文只把它们列为数学定位，不写成等价定理。

参考文献：

1. Sweldens, W. *The Lifting Scheme: A Custom-Design Construction of Biorthogonal Wavelets*. Applied and Computational Harmonic Analysis, 1996.
2. Dao, T., Gu, A., Eichhorn, M., Rudra, A., and Ré, C. *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*. ICML, 2019. arXiv:1903.05895.
3. Ebrahimi-Fard, K., and Rahm, L. *A Survey on the Munthe-Kaas-Wright Hopf Algebra*. 2023. arXiv:2306.04381.

## 13. 结论

TreeHeap 从一个关于有序树、旋转切片、有限容量和信息分辨率的直觉出发，经历了多次必要的失败。路径曾被误当成语义，张量可区分性曾被误当成正确结构选择，旧 checkpoint 曾出现严重方向坍缩，flat 表曾伪装成递归路由，几何特征也曾直接泄漏答案。正是这些失败，迫使最终系统明确区分地址与语义、参数与状态、root 与完整 (H)、数学闭包与归纳学习。

当前 TreeHeap 由一条完整链路组成：WRITE 把输入写入 leaf；共享可逆 kernel 按 XOR-Butterfly 调度建立稀疏全地址通信；lifting FOLD 把状态组织为 root 与多层 detail；递归概率 READ 在生成时选择不同分辨率；最终 token 交叉熵沿整条链反向传播，使 Encoder 与 Decoder 在没有人工内部标签的情况下共同适应。

正式三种子 WMT 结果支持 Butterfly 配置在当前训练合同中的稳定收益。早期 lifting 实验支持 root 与多个 detail 深度共同参与任务，可学习 Update 则说明梯度能够在保持可逆闭合时改变信息上导规则。全量双向训练进一步展示了一条不单调但可观察的行为轨迹：模型从高频重复发展为能够保留年份、事件和部分关系的源相关输出，同时仍受到重复、实体偏差和训练平台的限制。

因此，TreeHeap 已经从哲学直觉和 toy 公式发展成一个可训练、可逆、可干预、可恢复的算法对象。它尚未证明语义地址、真正压缩、工程效率或产品能力。下一阶段的任务不是继续扩大语言，而是用更严格的拓扑对照、最终质量指标和计算成本测量，判断这套结构究竟在哪些条件下提供独立价值。

## 附录 A：Claim 边界表

| 陈述 | 当前状态 |
|---|---|
| Butterfly 正反变换在训练后保持数值闭合 | 支持 |
| FOLD/UNFOLD 在训练后保持数值闭合 | 支持 |
| root 与多个 detail 深度参与当前 WMT 任务 | 支持机制 |
| 可学习 Update 能在保持闭合时改善当前任务 | 部分支持，未达到注册的 0.05 NLL 门槛 |
| Butterfly 配置在当前 WMT 合同中优于 Identity/Adjacent | 三种子支持 |
| 已训练模型依赖通信变换 | 运行时依赖检查支持 |
| 收益严格来自 changing-bit 拓扑 | 开放 |
| 一个共享 TreeHeap 正在形成中英双向协议 | smoke 与规模生长观察支持，最终质量开放 |
| root 是人类可读摘要 | 未证明 |
| XOR 地址天然具有语义 | 未证明 |
| 当前 TreeHeap 已实现存储压缩 | 否 |
| 当前 TreeHeap 已节省训练或生成计算 | 未证明 |
| 当前模型达到产品级翻译质量 | 否 |

## 附录 B：符号表

| 符号 | 含义 |
|---|---|
| (N) | TreeHeap 最大叶容量 |
| (D) | 最大递归深度，(D=\log_2N) |
| (m) | 单节点向量维度 |
| (d) | 一对 child 经 Predictor 得到的 lifting detail |
| (	heta) | 全数据共享、写入 checkpoint 的学习参数 |
| (H_\theta(x)) | 输入 (x) 产生的临时 TreeHeap 状态 |
| (r) | root 状态 |
| (d^{(k)}) | 第 (k) 层 lifting detail |
| (M) | 有效地址 mask |
| (F,G) | Butterfly 双节点耦合 kernel |
| (P,U) | lifting Predictor 与 Update |
| (S,B) | READ 的 stop 与 branch kernel |
| (h_t) | Decoder 在时间步 (t) 的隐状态 |
| (c_t) | recursive READ 得到的上下文 |

## 附录 C：最终稿待补项目

1. 等待 96 小时双向任务结束，冻结最好与最后 checkpoint；
2. 把完整远程 Evidence 同步回仓库并记录 SHA-256；
3. 统一 Native、Adjacent、Random topology 的验证样本；
4. 报告标准 BLEU、chrF、COMET、重复率和长度分桶；
5. 报告 en2zh 与 zh2en 的独立结果；
6. 测量训练 token、GPU 小时、峰值显存、吞吐和估算 FLOPs；
7. 增加三个以上固定全覆盖随机 topology 的多种子重训；
8. 将完整 dreams 轨迹作为补充材料发布，不只选择最好样例。
