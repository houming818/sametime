# SameTime / TreeHeap ARA 项目综述

状态：持续更新的研究入口

更新时间：2026-07-29

分支：`experiment/private-protocol-battle`

范围：M0 代数、S1 编码与 Echo、S2 翻译、S3 生成与索引

仓库说明：本文发布在 `main`；最新的 C14-C17 logic、代码与 evidence 目前
仍在 `experiment/private-protocol-battle`，等待正式集成评审。

这是一份能够单独恢复 TreeHeap 当前研究状态的入口文档。它集中说明：
项目到底在研究什么、哪些结论已有数据、哪些路线失败了、下一步应该验证
什么。它不能替代原始 evidence；若本文与 evidence 冲突，以 evidence 和
Claim 中注册的边界为准。

## 1. 一句话结论

TreeHeap 目前是一个**已有多项代数与因果机制证据、但还没有形成可用通用
语言模型的研究架构**。

现有证据支持以下有限结论：

1. 固定容量、有序地址的树堆可以实现一组有用的局部代数：组合、分解、
   路径差分、镜像、有限旋转、递归 FOLD/UNFOLD，以及概率化写入与搜索。
2. 在受控的迁移、路由和有限预算检索任务中，树地址、路径和子堆确实能
   提供可测量的归纳偏置。
3. 递归 encoder 可以让 root 和中间节点因果依赖左右子节点；被强制打开
   递归通道的 decoder 也能利用这些中间信息。
4. 无限预算下，精确条件计数的路径概率会望远镜消去，因此无法学习树的
   排列。只有加入内存、访问节点数、状态宽度或算力限制，拓扑才开始有用。
5. 数据可以先临时 STOP 在内部节点；随着写入压力增加，精确数据向下迁移，
   父节点保留折叠后的摘要。当前只有本地 smoke 支持碰撞分裂，还不是学会
   语言索引的 encoder。

现有证据**没有**证明语法自然涌现、语义世界模型、稳定的 encoder-decoder
私有协议、优于 Transformer、可用聊天能力或 GPT 级生成。当前自由生成较差
且经常重复，STONE-1 尚未完成。

## 2. 权威资料入口

| 要了解的内容 | 文件 |
|---|---|
| 根 Claim 树和研究历史 | [`PAPER.md`](PAPER.md) |
| 中文根论文 | [`PAPER.zh.md`](PAPER.zh.md) |
| M0 Claim 登记表 | [`m0-treeheap-math/logic/claims.md`](m0-treeheap-math/logic/claims.md) |
| S1 Claim 登记表 | [`s1-echo/logic/claims.md`](s1-echo/logic/claims.md) |
| S2 Claim 登记表 | [`s2-translation/logic/claims.md`](s2-translation/logic/claims.md) |
| S3 Claim 登记表 | [`s3-generation/logic/claims.md`](s3-generation/logic/claims.md) |
| 可复现代码 | `ara/*/src/` |
| 指标和运行摘要 | `ara/*/evidence/` |
| 失败路线和转向记录 | `ara/*/trace/` |

在 C14-C17 合入 `main` 之前，请到
[`experiment/private-protocol-battle`](https://github.com/houming818/sametime/tree/experiment/private-protocol-battle/ara/s3-generation)
查看对应的 logic、代码与 evidence。

ARA 的基本链条是：

```text
Claim -> Predict -> Experiment -> Evidence -> Decision -> Next Claim
声明      预测       实验          证据        判定          下一声明
```

Blog 用于帮助大众理解，不属于实验 evidence。`supported` 也只能支持 Claim
中明确写出的有限范围，不能自动升级成整个 TreeHeap 理论成立。

## 3. 项目到底在问什么

项目希望知道：固定容量、有序、可递归寻址的状态，能否学习 MLP、CNN、
Transformer 所处理的同类映射任务，并在以下问题上获得结构优势：

- 地址和路径；
- 子结构复用与迁移；
- 局部组合和分解；
- 有限搜索与前缀复用；
- 多分辨率信息；
- 延迟进行概率坍缩。

正确的比较不是“树和智能谁更强”，而是：

```text
相同任务 + 相同数据 + 匹配的内存和算力
-> TreeHeap 的结构偏置是否提高效率、因果性或外推能力？
```

## 4. 当前系统模型

```text
事件 / token / vector
        |
        v
WRITE 或 placement kernel（决定写到哪里）
        |
        v
有序 leaf 与内部节点的 STOP 数据
        |
        v
递归 FOLD / 信息抽水机
        |
        v
H_state：root + 带地址的中间 detail + leaf
        |
query + 递归 READ(stop, left, right)
        |
        v
概率容器
        |
在任务上下文中坍缩
        v
检索值、next token 或生成结构
```

Loss 可以训练 kernel 参数、token embedding、readout 参数；在 soft 版本中
也可以训练 placement 概率。几何或代数算子本身可以保持固定。当前代码并
不能支持“所有参数张量本身都是 TreeHeap”这个更强说法。

## 5. 核心术语

| 术语 | 当前严格含义 |
|---|---|
| TreeHeap | 固定容量、有序递归状态，具有稳定的父子地址和局部算子。仅仅把一个 tensor 摆成树形还不够。 |
| `H_state` | 运行时所有节点的内容，可以包括精确数据、摘要、detail 和隐向量。 |
| `theta` | WRITE、FOLD、route、UNFOLD 或 READ kernel 的可训练参数。必须和运行时 `H_state` 区分。 |
| Kernel | 在节点或子堆上反复应用的共享局部函数，可以输出状态更新、score map 或 `stop/left/right` 分布。 |
| FOLD | 从 child 到 parent 的算子。自底向上重复执行，形成 root 和多分辨率状态。 |
| UNFOLD | 从 parent 加 detail 还原 child 的算子。某些 codec 可以代数精确可逆，但可用语言生成仍是另一项归纳学习问题。 |
| 信息抽水机 | 递归 FOLD 将任务相关信息向上输送的过程。它是一种机制，不等于 root 已经是人类可读摘要。 |
| STOP | 可以出现在任何节点的合法结果。依模型不同，可表示精确数据、粗类别或概率桶。 |
| 概率容器 | 暂时不 argmax 的分布，例如 `{stop, left, right}` 或多个候选图。 |
| Mirror | 确定性的左右手性翻转。项目已统一不再称为“共轭”。 |
| 私有协议 | encoder 与 decoder 共同适配、但人类未必可读的编码规则。当前尚未证明可用于自由生成的有效私有协议。 |
| 分辨率 | root 附近感受野更大，leaf 附近局部细节更多。高层是否具有粗语义，仍需任务证据，不能由树形直接推出。 |

## 6. TreeHeap 证据必须满足的条件

只有确实使用了树结构，而非绕过它，实验才能算 TreeHeap evidence。

1. **固定容量**：明确节点数和每个节点的状态宽度。
2. **有序地址**：left/right 与路径身份必须参与计算。
3. **递归局部性**：共享的局部算子必须真正跨深度组合。
4. **没有隐藏 flat 直通**：decoder 不能偷看 target、使用无限 flat 表，或
   从输入特征直接拿到正确路由标签。
5. **守恒或明确损失**：迁移/FOLD 必须说明保存了什么、主动丢弃了什么。
6. **因果干预**：如果声称某个结构有用，shuffle、mirror、root/detail 消融
   或访问预算变化就必须影响结果。
7. **公平基线**：内存、访问节点、更新次数、数据和评价协议必须匹配。

## 7. 四层进度

| 层 | 目标 | 当前状态 |
|---|---|---|
| M0 | 建立代数工具箱 | 多项确定性恒等式和可微 toy 机制已有支持，是目前最牢固的地基。 |
| S1 | 写入、Echo、上下文路由 | 容量与受控路由成立；token-only 语义被否定；自然语料 encoder 和私有协议仍未解决。 |
| S2 | Fold Stack 与翻译 | Fold/action 信号存在，但 Graph Builder 和条件生成仍是瓶颈；历史翻译能力有限，且常落后 flat。 |
| S3 | 生成、codec、索引 | 多项因果机制成立，但自由生成和强私有协议不成立。C15-C17 已把前沿转向“有限预算条件索引 + 动态数据落位”。 |

## 8. 最强证据链

### 8.1 M0：代数和梯度入口

- 确定性 probe 支持闭包、非交换、类逆重建、投影、子堆匹配、mirror
  involution、有限域 decoder 和有限旋转。
- Diff 代数的有限差分绝对误差为 `4.21e-10`；一次梯度更新将 toy loss
  从 `29.1384` 降到 `0.00066`。
- Soft Plus 在注册 toy 中获得了非零梯度，并在低温下坍缩到正确 hard 地址。
  但它使用了人工 alignment 特征，不能升级成“已学会通用路由”。
- 全树 kernel convolution 可以表达确定性的 search/write/mirror score map。
  这证明了算子语言，不证明语言智能。

相关专业数学包括 rooted-tree algebra、Hopf/BCK 风格的组合与切割、operad
的 many-in/one-out 组合，以及经典 tree kernel。项目尚未证明完整的学习系统
严格等同于其中某一个成熟理论。

### 8.2 结构因果性

- 在受控 relocation toy 中，path + subheap 特征在 flat/path-only 失败时
  完成外推；证据范围仍是合成小实验。
- 四头 root compressor 的 valid NLL 为 `6.2365`；破坏左右地址配对使 NLL
  增加 `2.6252`，分别消融 head 也造成明显损失。这证明该模型利用了有序
  结构，不等于证明每个 head 有人类可读语义。
- 多分辨率 codec 形成了单调 rate-distortion 曲线；`k=64` 时 token top-1
  为 `0.9964`。这是有限重建机制，不是已经完成熵编码的语言压缩器。
- 冻结 encoder 后，强制递归读取将 test NLL 从 root control 的 `3.5149`
  改善到 `3.4636`；2% depth floor 又改善到 `3.4117`。这证明中间节点信息
  可以被读取，不证明模型会自然选择深度。

### 8.3 C14-C17：当前最前沿

**C14：target TreeHeap 历史状态。** 自回归 WRITE 和增量 FOLD 让 target
历史具有因果作用：清零历史使 NLL 增加 `4.3857`，只保留 target root 使
NLL 增加 `0.7718`。但生成仍不可用（BLEU-4 `0.2150`、unique output
`0.184`），source route 也完全坍缩到最深层。

**C15：有限预算条件索引。** 对精确条件计数：

```text
P(leaf | q) = 各层 P(child | q, node) 的乘积
            = C(q, leaf) / C(q, root)
```

中间节点被消掉，因此无限预算 exact NLL 与树排列无关。在最多访问 10 个
节点时，优化 placement 把 held-out Hit@3 从 `0.6921` 提高到 `0.8125`，
达到 flat exact Top-3。但 tree 使用的原始计数项更多，尚未证明压缩优势。

**C16：内部 STOP。** 允许精确数据停在任意内部节点，没有提高 Hit@3
（`0.8033` 对 `0.8033`），但把正确结果平均深度从 `4.0000` 缩短到
`2.2842`。这只是本地 smoke，不是正式 evidence。

**C17：压力分裂写入。** 固定 31 节点的 TreeHeap 把第一个数据写在 root；
第二个数据碰撞时，原数据向下迁移，parent 转为摘要。最终形成 16 个精确
leaf 数据和 15 个内部摘要。8 种插入顺序中，没有已有数据向上移动；signature
route 的预算 AUC 从随机的 `0.5269` 提高到 `0.5574`。但平均深度并非单调
增加，注册 gate 失败；signature 也是用共现计数人工构造的，并非端到端学会。

## 9. 关键失败与降级

- token-only path hash 证明的是容量，不是上下文语义。
- 历史 128D checkpoint 高度坍缩（cosine 约 `0.985`），不能证明语法能量
  问题已经解决。
- 朴素世界场/张量能量没有证明“正确句子天然最低能量”。
- 精确 count-path likelihood 会望远镜消去，不能训练树拓扑。
- 随机向量直接相加会丢失子堆身份，compose 必须是非平凡算子。
- root 因果有效不等于 decoder 递归读取。增加 50M 模型训练步数，也没有
  让它自然读取更深节点。
- C10 处理了 14.10 亿 target token，teacher-forced NLL 很低，但 CLI 坍缩
  成重复的“一带一路”。完整 target teacher forcing 和可见 EOS 使其失去
  翻译/私有协议证据资格。
- C11 证明了部分 source dependence，但仍重复短句。
- 一次性 H-state unfold（C12）和并行 emergent protocol（C13）均发生坍缩。
- teacher uncertainty 蒸馏没有优于 gold target。
- 单纯增加参数，没有改善注册的 50M 方案。
- 没有任何结果可以证明意识、感受、世界模型、GPT-2/GPT-4 能力或商业可用。

这些失败不是废料，而是防止项目再次绕回同一条死路的航标。

## 10. 已知与未知

### 已经知道

- TreeHeap 可以在多个深度携带精确和有损状态。
- 左右有序结构可以具有明确因果作用。
- 共享递归算子可以传递梯度与信息。
- 强制打开通道后，decoder 能利用中间节点。
- 有限搜索预算下，数据放置方式会改变检索质量。
- 内部 STOP 可以缩短访问路径。
- 固定内存中的碰撞压力可以让精确数据向下迁移。

### 仍然不知道

- 如何从自然语料中学习 placement，而不是人工设计 signature。
- 有限宽度节点应该保存什么，才能保留有用的条件分布。
- encoder 和 decoder 能否形成稳定、可用的私有协议。
- parent 是否会变成语义轮廓，而不是任意 hash。
- 在匹配算力和质量时，TreeHeap 能否超过强 Transformer、RNN、检索系统或
  压缩 flat 基线。
- 不使用强制 depth floor 时，变量深度路由能否自然形成。
- 固定容量 TreeHeap 能否生成流畅且与 source 有关的自由文本。

## 11. 下一项 Proof：概率桶的选择性下迁

下一步应当细化 C17，而不是立即再造一个 decoder。

假设一个内部节点的 STOP 概率桶目前是：

```text
depth1.node0 STOP -> {word1, word2, word3}
```

继续学习后，模型认为 `word3` 需要更细的区分：

```text
depth1.node0 STOP -> {word1, word2, word3} 的粗摘要
depth2.node0 STOP -> word1/word2 的精确组
depth2.node1 STOP -> word3 或它的更细概率桶
```

`word3` 不是全局遗忘。它的**精确记录向下移动**，同时它的概率质量或压缩
影响仍包含在 parent 的 subtree summary 中。必须注册守恒式：

```text
subtree_mass(parent, w)
  = local_mass(parent, w)
  + subtree_mass(left, w)
  + subtree_mass(right, w)
```

迁移必须原子化：先写 child、验证 child、重算 parent、验证质量/checksum，
最后才删除 parent 的精确副本。否则会悄悄重复或遗忘数据。

下一实验应在完全相同预算下比较：leaf-only、静态 internal STOP、随机 split、
人工 pressure split 和 learned pressure split。必须固定：

- 节点数与总 state byte；
- streaming event 顺序；
- `1..B` 节点访问预算；
- 参数更新算力；
- 不允许精确数据重复存储。

主指标包括：Hit@K-versus-visits AUC、完整检索准确率、parent mass 误差、迁移
遗忘率、内存 byte 和 held-out stream 表现。受控概率桶 proof 通过后，才进入
真实语料共现事件和 learned kernel。

## 12. 工程状态

- 当前评审仓库：`holds/SameTime-depth-growth`。
- GPU 主机：`io`；必须保持已有功率与频率保护，不能解除限制。
- 推荐通过串行 `taskd` 队列运行，直接 SSH 仅作为 fallback。
- `io` 本地语料：`/home/nio/datasets`，约 25 GB。
- NAS 源：`/mnt/nas/datasets`，约 51 GB。
- 数据包括 WMT 和中文新闻、网页、百科、BELLE、Baike、翻译、知乎及医疗语料。
- 正式远程实验必须保存 code commit、命令、环境、日志、必要 checkpoint 和
  `evidence/` 下的 `summary.json`。

## 13. 产品 Gate

STONE-1 尚未完成。要达到可发布产品里程碑，至少需要：

1. 生成确实依赖 source，而非 target teacher forcing 泄漏；
2. 低重复、具有有效 conditional diversity；
3. 多 seed 可复现；
4. TreeHeap 的地址与深度有因果作用，且不存在 flat bypass；
5. 与强基线公平比较，并记录训练 token、GPU 小时和算力；
6. CLI 声明的任务必须和训练 objective 一致；
7. 公开 checkpoint、tokenizer、推理命令、model card 和限制说明。

当前最接近诚实产品的方向，是有限容量 TreeHeap codec/index 研究 demo，而非
通用对话 AI。

## 14. 新评审者如何接手

1. 先读本文。
2. 找到对应 topic 的 `logic/claims.md` 行。
3. 阅读实验设计和源代码。
4. 检查 `evidence/*/summary.json` 与原始日志。
5. 确认干预和基线确实检验了 Claim 中的机制。
6. 同时更新 Claim 状态、trace、`PAPER.md` 与本文。

不要仅凭 blog 数量、训练时长、GPU 利用率、较低的 teacher-forced NLL 或
参数规模来判断项目进度。
