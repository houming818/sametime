# F03：TreeHeap 节点滤镜显微镜

日期：2026-09-08  
Claim：`S3-TREEHEAP-NODE-FILTER-MICROSCOPE-F03`  
状态：预注册，等待 smoke 与正式扫描。

## 1. 问题

固定训练完成的 TreeHeap、输入句和 Decoder，把运行态 TreeHeap 按 root 到 leaves 展开：

```text
H = [H_root, H_parent..., H_leaf...]
W = [w_0, w_1, ..., w_(n-1)]
H_filtered[i] = w_i * H[i]
```

这里的 `W` 是与节点一一对齐的滤镜，不是 F02 那种每个递归深度共享的 FOLD 标量。
滤镜位于原生 FOLD 完成以后、Decoder READ 以前，因此不改变 leaf 到 parent 的递归算法。
`W=1` 是严格原生读出；令某个 `w_i=0` 才是删除该节点的单坐标消融。

## 2. 固定标本

本轮只观察一个句子：

```text
id: sisyphus-hello-01
source: 西西弗每天都推着石头上山顶。
reference: Every day, Sisyphus pushes the stone to the top of the mountain.
```

人工语义槽只作辅助标记：实体、频率、动作、对象、方向和目的地。原始 Decode 文本是主证据，
不能因为字符分数或关键词命中而把错误译文判成正确。

## 3. 实验合同

- 基座：E01 TreeHeap-106M pass-2 checkpoint；
- depth 固定为 7，`K_up` 保持原生；
- checkpoint、Tokenizer、FOLD、Decoder 和采样规则全部冻结；
- TreeHeap 按 root 到 leaves、层内从左到右编号；
- 完整滤镜长度等于所有层的节点数，但只扫描当前标本 mask 有效的节点；
- greedy Decode，最多 64 pieces；
- 粗扫：每次只改变一个节点，`w=0.00..1.50`，步长 `0.05`；
- 精扫：按粗扫产生的不同输出数量排序，选择最多 8 个节点；对最靠近 `w=1` 的换相区间以
  `0.001` 步长确定性扫描；
- 不根据输出增删标本、改变网格或改写参考译文。

## 4. 记录量

每个坐标保存节点地址、层级、层宽、位置、权重、原始 token ids、原始文本、EOS、长度、
相邻重复率、character-F2 diagnostic 和语义槽命中。汇总另保存每个节点的不同输出数、
换相次数、首个换相区间以及精扫边界。

## 5. 预注册判定

### P0：身份滤镜合同

原生前向与 `W=1` 滤镜的 TreeHeap 张量逐元素相等，生成 token 逐个相等；checkpoint、代码、
标本哈希完整。失败则实验无效。

### P1：覆盖完整

所有有效节点都完成 31 档粗扫，地址无重复、无缺失，全部数值有限，结果文件可解析。

### P2：存在节点焦距效应

至少一个节点在粗扫中产生不同于 `W=1` 的原始 Decode token 序列。通过只说明该节点幅度是
当前冻结系统的运行态因果变量，不说明它独立承载某种语义。

### P3：效应跨分辨率

若 root、internal、leaf 三类中至少两类出现 Decode 换相，记录为跨分辨率滤镜效应；否则将
效应记录为局部化，不据此修改架构。

### P4：精扫解析换相

若粗扫存在相邻档输出不同，至少一个预定精扫区间应把边界收窄到 `0.001` 网格；否则记录为
当前直接 Decode 分辨率下未解析。

## 6. 边界

F03 是单 checkpoint、单句、单 depth 的显微镜实验。它不证明滤镜可泛化，不训练滤镜，
不把单节点幅度变化解释成向量旋转，也不授权替换默认 TreeHeap。它先回答更小的问题：
一棵固定 TreeHeap 的哪些节点，在什么权重附近，会让 Decoder 读出另一幅文本。
