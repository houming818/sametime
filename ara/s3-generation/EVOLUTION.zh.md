# TreeHeap 生成航线技术演化地图

更新日期：2026-08-24

## 0. 为什么需要这份地图

TreeHeap 的技术不是从 STONE-1 直接跳到 STONE-2，也不是依靠一组 `Cxx` 编号
自然长出来的。每次实验都在回答一个局部问题，成功结果成为构件，失败结果排除
错误结构。若只保留最终候选，读者会看不到设计为什么变成今天这样；若只保留
数字编号，读者又必须重读全部博客才能恢复上下文。

因此本文件使用三种标记：

- **航线阶段**：连续的技术演化，回答“上一阶段留下了什么问题”；
- **实验名称**：回答“这次到底改了什么”；
- **档案编号**：`Cxx`、任务号和 Claim ID，只负责定位 evidence，不表达先后或等级。

所有失败和勘误都保留。公开源代码不只意味着文件可下载，也意味着设计谱系可以
被理解和审核。

## 1. 数学工具阶段：先确认 TreeHeap 能做什么

### 1.1 从代数对象到卷积核

最早的工作建立 `plus`、差分、镜像、compose/decompose、递归遍历等基本算子。
这里区分两类结果：代数闭包属于演绎证明；概率 kernel 的参数学习属于归纳证据。

留下的构件：

- TreeHeap 状态可以被递归 kernel 遍历和更新；
- mirror 是左右地址的几何翻转，不再称为共轭；
- 可逆算子与可学习概率核必须分别证明。

对应公开过程：[SPR-022](https://www.grepcode.cn/spr/022-treeheap-math-foundation.html)、
[SPR-023](https://www.grepcode.cn/spr/023-treeheap-kernel-convolution-ops.html)、
[SPR-027](https://www.grepcode.cn/spr/027-treeheap-diff-algebra.html)、
[SPR-039](https://www.grepcode.cn/spr/039-parameter-treeheap-kernel-learning.html)、
[SPR-040](https://www.grepcode.cn/spr/040-mirror-kernel-symmetry.html)。

### 1.2 路由纠错：矩阵不是 TreeHeap

早期 route 使用 flat `L x L` 矩阵，随后递归版本又把目标区间直接写进 feature，
造成几何答案泄漏。审计后改为内容感知 route：kernel 必须读取当前节点和子节点
状态，不能只读预计算地址。

结果不是“route 已解决”，而是得到一个严格边界：

> 使用树形数组不等于使用 TreeHeap；状态、递归地址和 kernel 必须进入因果计算。

对应公开过程：[SPR-045](https://www.grepcode.cn/spr/045-recursive-treeheap-route.html)、
[SPR-046](https://www.grepcode.cn/spr/046-content-route-voyage-problem.html)。

## 2. Encoder 问题：从人工放置转向私有协议

### 2.1 世界观察者与可学习 compose

随机 token 向量相加无法保存 subheap 身份，说明问题不只是 Decoder 如何找路，
而是 Encoder 是否把可区分结构写进 `H_tree`。于是写入从“人工把词放到树上”
改为“由上下文损失训练 placement 和 compose kernel”。

这里经历过五种 loss 的过重设计，随后收缩为最小的 `echo + context`，并要求
shuffled corpus control。失败留下的约束是：compose kernel 必须参与学习，不能
先写 leaf 再被动求和。

对应公开过程：[SPR-047](https://www.grepcode.cn/spr/047-treeheap-encoder-world-observer.html)。

### 2.2 私有编码森林与语义前缀树

Encoder 和 Decoder 不需要使用人类可读的分类标签，但必须形成共同压缩协议。
单棵树可以表示一种观察方向，多棵参数树可以并行或串行工作；Huffman 类比只
说明路径压缩和概率坍缩，不意味着人工规定“食物/药品”节点。

留下的构件：

- 参数 `theta`、运行状态 `H`、查询 `Q` 和输出概率桶分离；
- 路径编码不等于输出概率；
- 私有协议可以不可读，但形成机制和因果使用必须可审核。

对应公开过程：[SPR-048](https://www.grepcode.cn/spr/048-treeheap-private-codec-forest.html)、
[SPR-049](https://www.grepcode.cn/spr/049-mask-kernel-structure-extraction.html)。

## 3. 从 Toy 到真实生成：WMT 暴露任务边界

### 3.1 第一次真实 seq2seq

WMT 让 TreeHeap 第一次生成像样的中文，证明真实序列训练可以跑通；但把翻译模型
直接问“地球为什么是圆的”失败，说明流利性不等于问答或世界知识。

留下的构件：

- 翻译、续写、问答必须分别声明任务；
- 生成像中文不等于形成逻辑；
- QA 需要相应数据和独立 Claim，不能改 CLI 标签冒充能力。

对应公开过程：[SPR-050](https://www.grepcode.cn/spr/050-real-wmt-seq2seq-voyage.html)、
[SPR-051](https://www.grepcode.cn/spr/051-p0-direct-qa-failure.html)。

### 3.2 Frontier 对照与算子选择

与平均、flat 等结构对照后，learned route 没有自动胜出。于是 Encoder 的职责从
“任意重写整棵树”收缩为“在 TreeHeap 代数算子集合中学习选择”。

留下的构件：固定代数负责闭包，梯度负责选择和组合；两者不能互相冒充。

对应公开过程：[SPR-052](https://www.grepcode.cn/spr/052-treeheap-frontier-bottleneck.html)、
[SPR-053](https://www.grepcode.cn/spr/053-treeheap-algebraic-operator-codec.html)。

## 4. 多分辨率形成：root 不再负责背诵全文

### 4.1 金字塔与多尺度 Mask

TreeHeap 被重新定义为分辨率金字塔：高层覆盖更大范围，低层保留细节。为了避免
模型只走 leaf 直通通道，Mask 从单 token 扩大到整段 subheap，让恢复任务必须
使用 parent 和 root。

这一步只部分支持“信息进入高层”，没有证明高层已经形成可命名的人类语义。

对应公开过程：[SPR-054](https://www.grepcode.cn/spr/054-treeheap-multiresolution-pyramid.html)、
[SPR-055](https://www.grepcode.cn/spr/055-treeheap-multiscale-mask-root-growth.html)。

### 4.2 差分观察与一次重要失败

层间差分被用作观察工具，但人工构造的类比差分得到 100% 后，因果干预证明模型
并未按声称机制工作。于是“差分直接找到意识/类比规律”的 Claim 被撤回。

留下的构件：差分可以当显微镜，不能当语义来源；漂亮准确率必须接受干预审计。

对应公开过程：[SPR-056](https://www.grepcode.cn/spr/056-treeheap-layer-differential-observation.html)、
[SPR-057](https://www.grepcode.cn/spr/057-difference-consciousness-failed-lesson.html)。

## 5. 可逆递归阶段：FOLD、detail residual 与 UNFOLD

### 5.1 Lifting Scheme

为了避免普通加权平均把信息抹掉，引入可逆 lifting：两个 child 被分解为 parent
轮廓与 detail residual；UNFOLD 可以演绎地恢复 child。随后把同一结构带入 WMT，
证明递归 READ 有结构因果信号，但翻译质量当时没有获胜。

留下的构件：

- `leaf -> parent + detail -> ... -> root` 是真正递归，不是多种长度的 flat 数组；
- 可逆 Echo 证明信息通道存在，但不证明语义协议优良；
- Decoder 应读取完整 `H_state`，不是要求 root 单独复原全文。

对应公开过程：[SPR-058](https://www.grepcode.cn/spr/058-treeheap-lifting-mask-echo.html)、
[SPR-059](https://www.grepcode.cn/spr/059-treeheap-lifting-wmt-recursive-read.html)。

### 5.2 退火、损伤修复与递归 Decoder

“抽水机”被改名并重述为退火压缩：高层容量下降，保留轮廓，detail 补充细节。
但上层不应被理想化为自然语言缩句；真正需要的是跨分辨率损伤修复和 top-down
递归读取。旧的“改变数组长度测深度”实验被判为测错对象。

对应公开过程：[SPR-060](https://www.grepcode.cn/spr/060-treeheap-annealed-contraction-protocol.html)、
[SPR-061](https://www.grepcode.cn/spr/061-treeheap-damage-repair-growth.html)、
[SPR-063](https://www.grepcode.cn/spr/063-treeheap-decoder-depth-growth.html)。

## 6. STONE-1：私有协议从机制证明走向可运行模型

### 6.1 私有协议、平台剂量与旋转

私有协议实验显示 TreeHeap 的结构参与了任务，但没有建立性能优势。随后发现
30K 数据平台会把架构差异压在 NLL 6.x 附近，因此把数据量、token 预算和参数
容量纳入实验合同。固定容量旋转被保留为观察方向变换，禁止指数扩张内存。

对应公开过程：[SPR-064](https://www.grepcode.cn/spr/064-treeheap-private-protocol-battle.html)、
[SPR-065](https://www.grepcode.cn/spr/065-treeheap-data-dose-platform.html)、
[SPR-066](https://www.grepcode.cn/spr/066-treeheap-rotation-private-protocol.html)。

### 6.2 Decoder 水压接通

STONE-1 的正式实验先发现 identity 优于 learned gate，说明 learned gate 没有学会
私有协议；随后冻结 Encoder，只训练 Decoder 的读取压力，证明多层状态可以对
输出产生因果作用。容量实验用于寻找率失真膝点，而不是假定参数越多必然越好。

对应公开过程：[SPR-067](https://www.grepcode.cn/spr/067-stone1-private-protocol-formal-result.html)、
[SPR-068](https://www.grepcode.cn/spr/068-treeheap-capacity-rate-distortion.html)、
[SPR-069](https://www.grepcode.cn/spr/069-stone1-encoder-decoder-pressure.html)。

### 6.3 第一个可下载候选与 C10 勘误

STONE-1 C08 成为第一个可以下载运行的翻译候选。之后更长的 C10 训练暴露固定
输出坍缩：loss 下降主要受训练目标和 EOS/数据分布影响，CLI 又曾错误标成翻译。
相关结论被公开撤回，没有用新的解释覆盖旧错误。

对应公开过程：[SPR-071](https://www.grepcode.cn/spr/071-stone1-candidate-c08-release.html)、
[SPR-073](https://www.grepcode.cn/spr/073-treeheap-coarse-to-fine-progressive-training.html)、
[SPR-074](https://www.grepcode.cn/spr/074-c10-loss-collapse-retraction.html)。

## 7. 长距离通信：Butterfly 进入主架构

二叉递归使远端 token 需要跨多层相遇。全树 Attention 会抹掉 TreeHeap 的局部
算子约束，循环位移又不能系统覆盖长距关系，因此引入固定容量 XOR Butterfly：
每轮仍是二节点 kernel，但按二进制地址交换，使有限深度内出现异构坐标折叠。

后续实验表明 Butterfly 有可测结构信号，但 runtime identity 并非严格消融，因为
它同时跳过了 Butterfly 参数。于是论文明确收缩了可声明边界。

对应公开过程：[SPR-076](https://www.grepcode.cn/spr/076-treeheap-butterfly-long-range.html)、
[SPR-077](https://www.grepcode.cn/spr/077-treeheap-paper-origin-and-evolution.html)、
[SPR-078](https://www.grepcode.cn/spr/078-treeheap-paper-math-and-dataflow.html)、
[SPR-079](https://www.grepcode.cn/spr/079-treeheap-paper-evidence-and-dreams.html)、
[SPR-080](https://www.grepcode.cn/spr/080-treeheap-paper-boundaries-and-reproduction.html)。

## 8. 视角协议：原序与异构折叠的比例/剂量

Butterfly 不是随机打乱，而是同一 string 的异构坐标折叠。问题从“是否使用
Butterfly”推进到“模型最终采用哪个视角作为输出协议”。通过原序 Identity 与
Butterfly 的四臂实验，比例和累计剂量被分开；JS、NLL、重复率和 Dreams 共同
显示，更多原序信号可以稳定主视角，但 NLL 与生成观感不总是同步。

留下的构件：预训练阶段和任务阶段可能需要不同 Identity 剂量，该趋势保留为待
验证假说，不能直接固化为架构定律。

对应公开过程：[SPR-081](https://www.grepcode.cn/spr/081-treeheap-private-protocol-viewpoint-drift.html)、
[SPR-082](https://www.grepcode.cn/spr/082-treeheap-canonical-view-dose.html)、
[SPR-083](https://www.grepcode.cn/spr/083-treeheap-stage-dependent-identity-dose.html)。

## 9. 有界退火与 READ 审计：STONE-2 的直接前史

### 9.1 FOLD 数值地基

退火 FOLD 被要求满足有界、不爆炸、不消失和可逆。参考零点与能量载体的 toy
审计发现：理论上的近奇异梯度风险存在，但真实 checkpoint 中并未出现相同病灶，
因此没有仅凭 toy 替换正式 FOLD。

对应公开过程：[SPR-085](https://www.grepcode.cn/spr/085-treeheap-fold-energy-and-gradient-pressure.html)。

### 9.2 STOP 坍缩与强制多层 READ

learned STOP 在训练中倾向把概率压到最细 leaf，使形式上的多层树退化为末层读取。
因此候选架构去掉 STOP，强制从 root 到 leaf 递归 READ。冻结审计显示 leaf-only
干预有明显损失，但“每个单独深度都不可替代”没有稳定成立。

对应公开过程：[SPR-084](https://www.grepcode.cn/spr/084-treeheap-c10-pretrain-stop-audit.html)。

### 9.3 能量载体解决了边界反例，没有解决原始层间干扰

这里必须保留完整的病因推理链，不能把一个成立的数学 Claim 直接写成原问题的
工程解答：

1. 原始观察是：关闭全部非 leaf READ 会让 Test NLL 增加约 `0.1311`，但单独
   删除任意一个深度反而略有改善。问题是多分辨率信息整体有用、局部却互相干扰；
2. 一个候选病因是参考系 FOLD 在强信号反向抵消时落入 epsilon 尺度，产生递归
   梯度放大；
3. 递归能量载体 `(direction, absolute_energy)` 的 toy Claim 成立：它把 alternating
   TreeHeap 的 root-to-leaf 梯度 norm 从 `70,710,677.94` 降到 `0.35355`，并保持
   FOLD/UNFOLD 与路径比例乘法精确闭合；
4. 真实 checkpoint 存在性审计却没有发现近抵消节点：全部节点的抵消指标都没有
   进入 `q<0.10`，全局最小值为 `0.58384`。能量载体还使长深度梯度衰减更快；
5. 因此，“能量载体能修复抵消型梯度爆炸”是受支持的数学 Claim，但“当前层间
   干扰由抵消型爆炸造成”被否定。算法保留为条件触发的数值工具，不替换正式 FOLD；
6. 原始层间干扰继续保持开放，不能登记为已解决。

后续真实任务梯度 Gram 把主要病灶定位到共享 READ kernel：coarse/middle 梯度
cosine 中位数为 `-0.1596`，负 cosine 比例为 `0.6667`，共享后的梯度抵消比中位数
为 `0.7003`；共享 branch 没有通过冲突门。这条证据把航线从 FOLD 数值修补移回
READ 私有协议，但它仍只是病因定位，不是最终修复。

因此当前状态应写成：

| 问题 | 当前判决 |
|---|---|
| FOLD 数值有界与可逆 | 参考系 FOLD 已有正证据 |
| 极端反向抵消导致梯度爆炸 | 能量载体 toy Claim 已支持 |
| 自然语言 checkpoint 是否存在该爆炸 | 当前审计不支持 |
| 多分辨率层间干扰 | 定位到共享 READ 冲突，尚未解决 |
| 稳定互补的多分辨率私有协议 | 尚未形成 |

## 10. 当前 STONE-2：不是四步跳跃，而是四个正在辨明的结构问题

### 10.1 统一训练管线

当前候选由以下历史构件组成：

```text
不可变数据 release
-> 自然文本 Pretrain
-> token WRITE
-> 固定容量 XOR Butterfly
-> 参考零点、有界、可逆 FOLD
-> 完整多层 H_state
-> 无 learned STOP 的强制递归 READ
-> recurrent Decoder
-> 翻译/问答 Task Train
-> 结构干预、生成、重载和 CLI 审计
```

组合 smoke 已通过 8/9 门；失败的是严格逐深度不可替代门。随后组级干预显示 coarse、
middle、fine 均有正的联合贡献，但存在负交互。这不是“树没参与”，也不是“多分辨率
已经解决”，而是定位到 READ 的参数共享和层间干扰。

档案入口：`logic/stone2_integrated_training_route.zh.md`。

### 10.2 共享 READ 的两个 successor

为解释层间干扰，先后验证：

1. **按 coarse/middle/fine 分组解绑定**：比完全共享略好，但没有击败同参数量的
   interleaved control，只支持“解除完全共享可能有益”，不支持人工三分组；
2. **连续深度低秩调制**：没有击败共享 READ，也没有击败打乱深度坐标的对照，
   当前参数化被否定。

这两个都是探路分支，不是新的 STONE 等级，也没有从历史中删除。

档案入口：`logic/stone2_grouped_read_matched_smoke.zh.md`、
`logic/stone2_continuous_depth_read_c05.zh.md`。

### 10.3 正在运行的稳定性复制

当前长任务验证**强制多层 READ 是否跨 seed 稳定**。它比较 learned STOP、强制
多层 READ、以及多层 READ 加第二次 upward kernel；保持父 checkpoint、数据流、
训练预算和评价完全匹配。

它回答的是机制稳定性，不是一步完成 STONE-2 产品训练。即使逐深度门再次失败，
也会留下“整体多层状态有用，但单深度不可替代不是正确分解方式”的稳定负证据。

档案入口：`logic/multilevel_read_ablation_c12.md`、
`logic/multilevel_read_ablation_c12r1.zh.md`。

## 11. 编号翻译表

下面的编号只用于定位，不要求读者记忆：

| 档案编号 | 本文件中的人话名称 | 作用 |
|---|---|---|
| STONE-1 C08 | 第一个可下载候选 | 产品化入口 |
| STONE-1 C10 | 全语料长训与坍缩勘误 | 暴露目标/CLI/STOP 问题 |
| C12 / C12-R1 | 强制多层 READ 稳定性实验 | 隔离 READ 与额外 upward kernel |
| C13 | 有界参考系退火 FOLD | 数值地基候选 |
| STONE-2 C03 | 统一管线组合 smoke | 检查完整计算图 |
| C03-D02 | coarse/middle/fine 联合贡献诊断 | 发现层间负交互 |
| STONE-2 C04 | READ 参数解绑定实验 | 否定人工三分组 |
| STONE-2 C05 | 连续深度调制实验 | 否定当前低秩线性调制 |

编号看起来不按时间排序，是因为它们来自不同局部实验系列。此后所有报告必须先写
人话名称，再在括号中附档案编号。

## 12. 当前真正的下一阶梯

下一阶梯不是“把所有指标一次做完”，也不是“再改一个系数”。它应同时包含：

1. 一个足够实质的结构变化：让多层 `H_state` 形成稳定、互补而非相互冲刷的 READ；
2. 一个明确能力结果：匹配预算下，生成或 held-out 任务质量不因结构约束而系统退化；
3. 一个结构证据：整体多层读取的因果作用跨 seed 存在；
4. 一个诚实边界：不要求每个深度都单独不可替代，也不把一次 NLL 改善写成完整私有协议。

当前长任务结束后，应根据跨 seed 结果决定下一种 READ 原理，而不是按 `Cxx` 顺序
机械增加编号。
