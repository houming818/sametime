# SPR-S2-G: 语义前缀路由 (Semantic Prefix Routing) 理论与架构总纲

> **说明**：此文档作为我们沟通和编辑的“同一大文档”。我们将在这里确立所有理论基础、数学建模和数据结构设计。理论达成共识后，再严格按照此文档的规约进行编码。

## 1. 核心目标与哲学 (Core Objectives)

### 1.1 破除“平坦世界”的参数隔离诅咒
传统的 NLP 模型使用 `nn.Embedding(V, D)` 作为词汇表映射。这种结构在拓扑上是“扁平（Flat）”的，每个 Token（如“猫”和“狗”）对应独立的参数行。
- **痛点**：当我们用平行语料（如中英文对齐）训练“cat-猫”时，梯度只能更新“猫”和“cat”的参数，无法跨 Token 扩散到“狗”或“dog”。
- **目标**：在缺乏海量平行语料（零样本/少样本）的场景下，实现**语义的自然泛化与推理**。

### 1.2 拓扑升维：树状结构先验
引入 **堆树（Heap Tree）** 作为词表的新型数据结构。
- 将每个词表示为从树根到叶子的一条**路径（Path）**，而非一个孤立的点。
- 两个词在树中共享的祖先节点越多，它们的语义联系越紧密。
- **动态泛化**：对一个节点（锚点）的更新，会自动通过共享路径扩散到具有相同前缀的所有子节点（非锚点词）。

## 2. 数学模型与算法设计 (Mathematical Modeling)

### 2.1 堆树路由寻址 (Heap Tree Routing)
- **结构定义**：采用满二叉树（或多叉树），深度设定为 $L$（如 $L=5$，共 31 个节点）。
- **硬路由公式**：对于词表大小 $V$，Token ID $i$ 的路径通过确定性的二进制规律分配：
  $$ \text{Node}_l = (i // (V / 2^l)) \pmod{2^l} $$
  *(注：具体路由分配公式可根据平衡性微调，核心是确保前缀共享)*
- **多叶性与多义词**：传统的一字一义会导致路径被独占。为了支持多义，模型需允许上下文（Context）动态参与子树（Branch）的权重路由，即从“静态单一路径”演进为“上下文加权路径”。

### 2.2 特征聚合：复平面上的几何变换
- **为什么不用加法/多层 MLP？** 深层递归和非线性 MLP 会带来噪音，导致优化坍塌（深度递归噪音 > 信号）。**[2026-06-04 编者注：此处理论存疑，不一定绝对正确，深层递归的失败可能受到具体超参和初始化影响，需要在后续工程中设计对比实验重新验证。]**
- **复数乘法（Complex Multiply）**：采用单层复数乘法。
  - 节点向量被表征在复平面 $\mathbb{C}^D$ 上。
  - 父节点与子节点的组合，在数学上等价于**相角的旋转（Rotation，改变语义指向）**和**模长的放缩（Scaling，改变语义强度）**。
  - 这完美契合了语言学中的“词根 + 词缀/修饰语”的组合逻辑。

### 2.3 优化目标：InfoNCE (对比学习)
- **摒弃 MSE**：均方误差（MSE）试图强行拟合绝对坐标点，这破坏了高维流形的几何特征，导致维度坍塌（如历史实验中 `cos` 降至 0.08）。
- **采用 InfoNCE**：
  $$ \mathcal{L} = -\log \frac{\exp(\text{sim}(u, v^+) / \tau)}{\sum \exp(\text{sim}(u, v_j) / \tau)} $$
  - 只优化锚点（跨语言平行词对）之间的**相对角度（Cosine Similarity）**。
  - 维持空间的内在几何，保留非锚点词汇相对位置的合理性。

## 3. 评测与基准 (Robust Evaluation)

### 3.1 BPE Token 的均值聚合 (Mean Embedding)
- **避坑（e[0] Bug）**：单字或单 BPE Token（如经常出现的占位符 `▁`）不能代表一个完整的词意。
- **评测策略**：任何词级别的语义相似度评测，必须将该词拆解为所有 BPE Token，并在流形空间上求均值（Mean Embedding），然后计算余弦相似度。

### 3.2 黄金锚点库 (Golden Anchors)
- 放弃纯位置统计（平行语料对齐错误率高）。
- 引入强模型（如 LaBSE）作为教师，筛选出 `cos > 0.35` 的高质量词对作为训练锚点（Train Anchors）。
- 单独剥离纯手工标注的词对作为黄金评测集（Gold Eval Anchors），严禁泄露到训练集中。

## 4. 阶段性实验策略 (Phased Experimental Strategy)

你抓到了极为关键的逻辑基石：**没有自编码器（Autoencoder），就没有跨语言编码器（Cross-lingual Encoder）。** 
如果堆树（Heap Tree）的路径路由和复数乘法连自身的 Token 都无法无损地表达和还原（Echo），那就根本无从谈起将两种不同语言的流形对齐。因此，工程实施将分为两个严谨的阶段：

### 阶段一：Echo 验证阶段 (Self-Reconstruction) - [已完成]
- **目标**：验证 Heap Tree 的表征容量。输入 Token $X$，经过 Tree 路由成向量 $E$，再通过简单的解码器（如一层 Linear）还原出 Token $X$。
- **矩阵测试结果 (2026-06-04)**：
  我们完整运行了 $3 \times 3 \times 3$ 的超参数矩阵测试（Depth $L \in \{3, 5, 7\}$, Dimension $D \in \{64, 128, 256\}$, Aggregation $\in \{\text{complex\_mul}, \text{simple\_add}, \text{mlp\_add}\}$），发现：
  1. **`complex_mul` (单层复数乘法)**：在所有深度和维度组合中均实现了完美的 $100.0\%$ 完美重构，且由于其不含深层非线性映射，收敛速度极快（平均每组运行仅需 $1.5\sim2.0$ 秒）。这彻底证实了复平面几何变换（旋转与缩放）在保留深层递归身份信号上的卓越拓扑优势。
  2. **`mlp_add` (相加后过 MLP)**：虽然也能达到 $>99.9\%$ 的重构精度，但训练开销较大（平均运行需 $3.5\sim5.0$ 秒），且容易受到冗余参数和过拟合的干扰。
  3. **`simple_add` (向量直接相加)**：在深度增加或维度较低时表现出灾难性的性能坍塌。例如，在 $L=7, D=64$ 时重构率仅为 $1.36\%$；在 $L=7, D=256$ 时也仅有 $31.46\%$。这用确凿的实验数据证明了“直接向量叠加会使深层递归身份信号被稀释和淹没”的假说，排除了线性相加架构。
- **结论**：后续对齐实验全部采用 **`complex_mul` (单层复数乘法)** 架构。

### 阶段二：Translation 对齐阶段 (Cross-lingual Alignment) - [进行中]
- **目标**：在 Echo 证明树结构具备充足容量的基础上，引入 LaBSE 锚点，使用 InfoNCE 进行中英跨语言空间的相对位置对齐。
- **产出模块**：
  1. **`spr_dataset.py`**：负责联合 BPE 模型的加载，处理 BPE Mean Embedding，加载 LaBSE 1428 组高质量锚点作为训练集，并将手工 1000 组金标准锚点严格隔离开作为测试集，提供高效的 padded batch 生成。
  2. **`spr_model.py`**：核心对齐模型 `SemanticPrefixRoutingModel`，实现对多 BPE Token 进行均值聚合后经过单层复数乘法与 `t_merge` 恒等初始映射，计算英-中双向 InfoNCE 损失。
  3. **`spr_eval.py`**：执行 BLI P@1（Bilingual Lexicon Induction）评测，对未参与训练的手工金标准集进行 1-to-1 匹配精度的计算。
  4. **`spr_trainer.py`**：全量 InfoNCE 对齐训练流程，支持从 Phase 1 预训练 Echo 进行 L0 暖启动。
  5. **`spr_align_matrix_inner.py`**：在单独的 Docker/Screen 会话中一键调度运行 Phase 2 的 3×3×3 网格搜索实验。
- **矩阵测试进展 (2026-06-04)**：
  我们已正式发射对齐阶段 27 组大矩阵运行（Depth $L \in \{3, 5, 7\}$, Dim $D \in \{64, 128, 256\}$, Aggregation $\in \{\text{complex\_mul}, \text{simple\_add}, \text{mlp\_add}\}$）。各组实验以每 Epoch 150 Steps、共计 50 Epochs 的规格在 GPU (cuda) 容器中执行，正由 `align_matrix_full_run.log` 进行全量跟踪。

## 5. 工程实施管线 (Implementation Pipeline)

基于以上理论，工程（S2-G 系列）的代码将严格按照以下模块构建：
1. **`spr_dataset.py`**：负责 Tokenizer、BPE 处理、LaBSE 高质量锚点对的加载与 DataLoader 构建。
2. **`spr_tree_layer.py`**：实现 Heap Tree 结构、复数乘法聚合机制。
3. **`spr_model.py`**：组装 Tree Layer 和 InfoNCE 损失函数计算。
4. **`spr_trainer.py`**：训练循环、早停（Early Stopping）逻辑以及 TensorBoard / Log 记录。
5. **`spr_eval.py`**：严格执行 BPE Mean Embedding 和 Gold Eval 计算。
6. **实验运行环境集成 (`q.py`)**：实验基于特定的设备 IO 与调度环境。我们需采用当前环境下的任务管理小脚本 `/home/nio/log/holds/SameTime/benchmark/wmt/q.py`，该脚本负责分配 GPU 资源和监控训练状态。新编写的 Python 入口必须兼容该脚本的调用/调度规范。

---
## 6. L0, L1, L2 渲染与翻译架构层级设计 (L0, L1, L2 Rendering Architecture)

为了支持下一阶段 Phase 3 的文本生成解码与路由对齐，我们将翻译流形解构为三个互不重叠、职责清晰的渲染层：

### 1) L0 层：Token 直译 (Lexical Lookup & Translation)
- **定位**：无上下文、单纯的 Token 到 Token 的词汇级映射。
- **机制**：通过 L0 词表嵌入查表实现，对应最核心的“静止语义质心”，是 Echo 重构的核心底座。

### 2) L1 层：单词渲染 (Contextual Word Disambiguation)
- **定位**：解决多义词歧义，结合周围邻近词对当前词的路径路由进行引流和投影旋转。
- **机制**：例如当 English 出现 "river" 邻近词时，其上下文在 Heap Tree 极坐标系中旋转/拉引 "bank" 的向量，使其精确偏转到“河岸”而不是“银行”的目标流形象限。

### 3) L2 层：语序构建与语法特征渲染 (Syntax & Target Order Rendering)
- **定位**：构建目标语言的语序，渲染目标语言的特定语法特征（如定冠词 "the"、不定冠词 "a" 的插入渲染以及语法格/时态）。
- **机制**：处理语序重排与结构词补齐，将 L1 生成的消歧语义流，重新排版并渲染出完整的符合目标语言文法的文本流。

---
*(等待编辑与审阅)*