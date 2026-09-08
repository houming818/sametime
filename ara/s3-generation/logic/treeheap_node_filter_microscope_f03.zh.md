# F03：TreeHeap 节点滤镜显微镜

日期：2026-09-08  
Claim：`S3-TREEHEAP-NODE-FILTER-MICROSCOPE-F03`  
状态：smoke 与正式扫描完成；P0--P4 全部通过，但没有恢复完整标本语义。

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

## 7. 实验结果

### 7.1 合同与规模

Smoke taskd `373` 完成 12 次代表性 Decode，身份滤镜与覆盖门通过。正式 taskd `376` 用时
`114.00` 秒，完成全部 `775 = 25 * 31` 次粗扫和 `408 = 8 * 51` 次精扫。

运行态完整 TreeHeap 的层宽为：

```text
[1, 2, 4, 8, 16, 32]
```

完整滤镜有 63 个坐标。当前句子的 leaf budget 为 12，mask 有效节点按层累计为 25 个：
root 1 个、internal 12 个、leaf 12 个。所有 25 个坐标在 `w=1` 时都逐 token 复现原生输出。

原生输出是：

```text
The whole of the day of the week, the chambers of the air.
```

它没有命中预注册的六个语义槽，因此这个标本的起点本来就是失焦译文。

### 7.2 节点响应

25 个有效节点中，19 个在 `0..1.5` 粗扫里至少产生一次不同 token 序列：

| 分辨率带 | 有效节点 | 发生换相 |
|---|---:|---:|
| root | 1 | 1 |
| internal | 12 | 9 |
| leaf | 12 | 9 |

全部粗扫合计出现 43 种不同 token 序列。最敏感的是 leaf `index=34`，31 档中出现 10 种
输出和 9 次相邻换相；其次是 leaf `index=35/37`，各出现 6 种输出。节点响应明显不均匀，
但不是只有某一分辨率带有效。

一个值得继续观察的例子来自 internal `index=3`、`level=2`、`position=0`：

```text
w = 0.20..0.40
The whole of the day of the week is a stone's throwing from the top of the hill.
```

它相对原生输出引出了 `stone` 和 `top`，但仍缺少 `Sisyphus`、`push` 和准确的频率关系。
这只能记作语义偏转，不能记作翻译成功。粗扫的 775 个单元中，671 个不命中语义槽，99 个
命中一个，只有 5 个命中两个；没有输出恢复完整六槽语义。

### 7.3 精扫边界

八个预定敏感节点都在 0.001 网格上解析出离散 Decode 边界。部分边界如下：

| 节点 | 边界括区间 | 边界后的可见变化 |
|---|---:|---|
| root `0` | `[0.674, 0.675]` | `...rush of the mountain` 回到原生输出 |
| internal `3` | `[1.396, 1.397]` | 转为重复 `chambers` 的句子 |
| internal `8` | `[1.364, 1.365]` | 出现 `stroll from the top of the hill` |
| leaf `34` | `[1.247, 1.248]` | `day of the day` 重复出现 |
| leaf `35` | `[1.177, 1.178]` | 出现 `mountainous Peak` |
| leaf `37` | `[1.246, 1.247]` | 出现 `mountainous Peak`，同时产生错误内容 |

这些边界不是连续语义距离的直接测量。它们是连续 hidden-state 幅度经过离散 greedy
argmax 和 EOS 决策以后表现出的换相点。边界可复现并不表示边界两侧的语义变化也连续。

### 7.4 判定

| 门 | 结果 | 证据 |
|---|---|---|
| P0 身份滤镜 | 通过 | TreeHeap 张量逐元素相等，原生与全 1 滤镜 token 完全一致 |
| P1 完整覆盖 | 通过 | 25 个有效节点、31 档、775 条粗扫全部完成且数值有限 |
| P2 节点焦距效应 | 通过 | 19/25 节点产生不同于原生的直接 Decode |
| P3 跨分辨率效应 | 通过 | root、internal、leaf 三类都发生换相 |
| P4 精扫解析 | 通过 | 8/8 预定区间在 0.001 网格内解析出边界 |

GPU 采样保持 270 W 功率限制，峰值功耗约 169 W、峰值温度 65 C、峰值显存 870 MiB；
日志中没有 OOM、CUDA/Xid、NaN/Inf。正式 evidence 位于：

```text
ara/s3-generation/evidence/s3_treeheap_node_filter_microscope_f03/formal/
```

### 7.5 当前结论

F03 支持：**逐节点幅度数组确实构成一个可操作、跨分辨率、具有不同敏感度的 TreeHeap
读出滤镜。** 它比 F02 的三个全局标量提供了更细的因果坐标，也显示同一输出附近存在
可解析的离散换相边界。

F03 不支持：**手工单坐标扫描已经找到正确焦距。** 当前最好看的偏转只恢复了部分
`stone/top/mountain` 信息，没有恢复人物和动作。下一阶梯如果训练 `W`，应先加入多句标本、
身份邻域正则和冻结基座审计；不能从这一个句子的扫描结果直接给 63 个坐标指定固定权重。

## 8. `push` 重复放大探针

为了区分“模型没有 `push`”与“组合句中 `push` 没有显影”，taskd `378` 使用同一冻结
checkpoint，在不加节点滤镜的条件下做了四档中译英直接 Decode。所有输入均未超过 32 pieces：

| 输入 | depth 5 | depth 6/7 的共同现象 |
|---|---|---|
| `推。` | 没有生成 `push` | 没有生成 `push`，输出失真 |
| `推` 重复 8 次 | `For more, pushing, push, and push.` | 明确生成 `pushing, push...`，随后重复到 64 pieces |
| `推石头` 重复 4 次 | `The stones of the stone was found.` | 生成 `stone`，没有 `push` |
| `西西弗推着石头上山` 重复 3 次 | 生成 `west/hill` | 生成 `stone/west/hill`，没有 `push` |

因此，“当前压缩空间完全没有可生成的 `push`”被这个放大探针否定。更窄且与观察一致的
工作 Claim 是：

> 冻结模型具有从孤立、重复的“推”到 `push/pushing` 的可读通路；但当动作与对象、人物、
> 方向组成结构后，当前压缩与 READ 没有让动作信号和其他语义同时显影。

重复不是正确翻译能力的证明。depth 6/7 在强重复输入下无法及时 EOS，说明它只是把动作信号
推过了生成阈值，同时造成了严重的重复失稳。这个结果把下一问题从“词是否存在”收窄为
“组合压缩为什么选择 `stone/hill`，却抑制了 `push`”。
