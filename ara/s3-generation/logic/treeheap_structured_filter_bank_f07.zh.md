# S3-TREEHEAP-STRUCTURED-FILTER-BANK-F07：结构化解析滤镜显微镜

状态：冻结模型 smoke 已完成；测量合同通过，结构响应各向异性得到支持，语义恢复未得到支持。

## 1. 问题

F06 证明固定解析滤镜能够改变 TreeHeap 的 FOLD、卷积树、READ 路由和 Decoder logits，但其
`pascal-middle` 实际按路径右转次数取 `C(d,r)`，同时混合了路径平衡与深度效应。本实验不要求
滤镜严格复现某个经典数列，而把滤镜视为已知形状的探针，分别观察整树位置、尺度和拓扑。

## 2. 冻结合同

- 沿用 F06 的 106M checkpoint、F04 theta 和“西西弗每天都推着石头上山顶”固定标本；
- 模型和 theta 完全冻结，不计算梯度，不做 target 优化；
- 固定宽度 32，base/extra 分通道干预；
- 扫描 `epsilon = +/-0.01, +/-0.1`；
- 保存实际滤镜数组、逐层 FOLD/卷积变化、READ 路由、固定历史 logits、自由生成和 token 频谱。

## 3. 滤镜组

### 3.1 能量滤镜

能量滤镜保持非负，只做单位 RMS，不减均值：

1. `uniform-gain`：全坐标等增益；
2. `pascal-row-energy`：对 n 个有效坐标使用长度 n 的二项系数行；
3. `gaussian-center-energy`：有效坐标序列中心的平滑聚焦；
4. `impulse-center-energy`：中心单点脉冲；
5. `ancestor-chain-energy`：最细层中心节点及其有效祖先链；
6. `subtree-energy`：中间 parent 及其有效后代子树。

它们回答“向何处增加能量会改变什么”。

### 3.2 重分配滤镜

重分配滤镜在有效坐标上减均值后做单位 RMS：

1. `triangle-redistribute`：中心增益、边缘抑制的线性窗；
2. `center-surround`：中心正、邻域负的 Mexican-hat 形探针；
3. `alternating-highpass`：相邻坐标正负交替；
4. `depth-fine-minus-coarse`：fine 与 coarse 尺度重分配；
5. `haar-local-sibling`：同层相邻 parent 的局部差分。

它们回答“固定总体尺度时，重新分配能量会改变什么”。两类滤镜不得用同一种零均值合同解释。

## 4. 预注册观察

- 比较相同 RMS、相同 epsilon 下各滤镜的 FOLD、卷积和 logits L2 响应；
- 检查扰动是否沿祖先链、子树或相邻 sibling 的预定拓扑传播；
- 比较连续响应与自由生成离散变化，不把单次文本变好写成质量提升；
- 记录 `push` 等固定概念的概率覆盖与 rank，但不以目标词出现作为成功门。

## 5. 完整性门

- `A0`：reader 与原路径一致，模型和 theta bitwise 冻结；
- `A1`：能量滤镜 RMS 为 1；重分配滤镜均值为 0 且 RMS 为 1；
- `A2`：全部单滤镜、双通道、双符号、双强度结果有限且完整；
- `A3`：四组预注册干涉结果完整；
- `A4`：所有参数 `requires_grad=False`，不存在 target optimization。

这些门只证明测量合同成立。滤镜响应不等价于语义坐标、质量提升或可训练路由已经成立。

## 6. Smoke 结果

运行：`io` taskd `397`，seed `11801`，耗时 `38.36 s`。`A0-A4` 全部通过；模型与 theta
保持冻结，base/extra 均完成 11 个滤镜、4 个 epsilon 和预注册干涉对。

在 base 通道、`epsilon=+0.1` 下：

| 滤镜 | FOLD L2 | logits L2 | route max JS | branch flips |
|---|---:|---:|---:|---:|
| uniform | 8.959 | 59.957 | 0.000542 | 4 |
| ancestor chain | 7.927 | 53.404 | 0.000685 | 5 |
| Gaussian center | 6.844 | 52.277 | 0.000393 | 4 |
| Pascal row | 5.811 | 45.152 | 0.000353 | 5 |
| subtree | 5.902 | 43.695 | 0.000679 | 2 |
| alternating high-pass | 3.488 | 42.836 | **0.000886** | **11** |
| sibling Haar | 3.609 | 35.282 | 0.000588 | 7 |
| depth fine-coarse | 4.812 | 32.730 | 0.000065 | 2 |
| center impulse | 3.650 | 25.233 | 0.000393 | 6 |
| triangle redistribution | 3.524 | 23.000 | 0.000164 | 6 |
| center-surround | 3.056 | 19.063 | 0.000133 | 6 |

能量滤镜含 DC 分量，不能仅凭响应较大判定形状更有效。更重要的观察是 alternating high-pass：
其 FOLD L2 不到 uniform 的 40%，但 route JS 和 branch flips 均为最高，说明 READ 对相邻坐标的
正负重排比对平滑总能量更敏感。ancestor-chain 与 subtree 也产生高于多数整树重分配滤镜的
route JS，支持继续用真实拓扑路径做显微镜扫描。

extra 通道的 FOLD/logits 响应整体远低于 base。例如 uniform 的 logits L2 仅 `7.369`，而 base
为 `59.957`。extra 中较多 branch flip 同时伴随很小的 JS，可能是近似平局的离散翻转，不能
单独解释为强路由变化。

## 7. 语义边界

基线自由生成仍为错误译文：

> The Westin is a mountainous stone, which is a stone's throw from the mountain of the hillside of

滤镜能够在若干生成模式之间切换，但没有恢复“西西弗推石头”的正确句子。`push` 的基线
phrase coverage 为 `0.0013051`；本轮最高为 base sibling Haar、`epsilon=-0.1` 的
`0.0013708`，单 token rank 从 `438` 改为 `423`，但生成文本仍没有出现 push。故本轮支持
“解析滤镜可以区分 TreeHeap 的敏感方向”，不支持“已经找到目标语义滤镜”。
