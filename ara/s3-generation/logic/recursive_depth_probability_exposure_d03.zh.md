# D03：递归深度概率场暴露探针

日期：2026-08-26

Claim：`S3-RECURSIVE-DEPTH-PROBABILITY-EXPOSURE-D03`

状态：`supported frozen recursive exposure`；taskd `317` 完成。

## 1. 问题

C12 与 C12-R1 已经观察到：把全部非 leaf READ 一次性关闭会明显恶化 NLL，
但从完整 READ 中单独删除某个深度，往往略微改善 NLL。这说明“删除一个部件”
不适合描述递归算法的形成过程。

本实验改用正向观察：固定同一个训练完成的 checkpoint，从 root 开始，依次允许
READ 执行到深度 `0, 1, ..., D`。深度 `d` 表示局部 TreeHeap READ 已经递归执行
了多少层，而不是人为指定的 coarse/fine 标签。

Butterfly 保持 Native 混合坐标，不执行 inverse Butterfly。FOLD、模型参数、
target、teacher-forcing 历史和输出层全部冻结。

## 2. 数据流

对每一个 target step，分别计算：

```text
root only                     -> logits_0 -> probability_0
root + depth 1               -> logits_1 -> probability_1
...
root + ... + leaf            -> logits_D -> probability_D
```

当前 READ 的累计形式为：

```text
q_(d+1) = q_d + gain * K_read(q_d, local_d, depth_d)
```

深度上限为 `d` 时，context 使用该深度的 `local_d` 与从 root 到 `d` 的累计
query residual。它不是完整状态删除一层，也不是把 TreeHeap 改成多个长度数组。

## 3. 冻结对象

复用 C12/C12-R1 的三个 `read` checkpoint：

- seed `10101`；
- seed `10102`；
- seed `10103`。

三个 seed 使用相同 WMT test row hash。正式探针固定前 256 条 test rows；smoke
只用 seed `10101` 的 32 条。探针没有 optimizer、没有 backward、没有参数更新。

不同长度句子的实际 TreeHeap 深度可以不同。若某条句子在累计深度 `d` 之前已经
到达 leaf，则后续深度保持其完整 READ 结果，不虚构新的节点；报告同时记录每个
累计深度仍对应真实节点层、而非 leaf 饱和复制的 token 比例 `active_fraction`，
避免把短树饱和误读为深层无效。

## 4. 逐层记录

每个累计深度记录：

- teacher-forced Test NLL 与 PPL；
- 词表概率熵；
- 正确 token 的平均概率与 top-1 命中率；
- 与前一个累计深度之间的平均 JS 距离；
- 正确 token 对数概率得到改善的 token 比例；
- top-1 token 改变比例；
- 当前 frontier 地址熵；
- 该累计深度仍对应真实节点层的 token 比例；
- 固定样本的 teacher-forced top-1 文本与若干位置 top-5 概率。

同时运行四个冻结条件：

1. `native`：原生 Butterfly 与原生 FOLD；
2. `runtime_identity`：Butterfly runtime 改为 identity；
3. `pair_break_depth_0`：破坏第一层 FOLD 配对；
4. `source_shuffle`：错配 batch 内 source。

后三个条件不是替代架构，只用于判断累计深度曲线是否依赖输入与 TreeHeap 结构。

## 5. Predict

### P0：证据合同

- 三个 checkpoint state SHA-256 与内部记录一致；
- 三个 seed 的 test row SHA-256 一致；
- 所有 logits、概率与统计量有限；
- 在完全相同的 test rows 与 batch 上，累计 READ 开放到完整深度后，
  logits 与原生 READ 的最大绝对误差不超过 `1e-6`，NLL 差不超过 `1e-6`。

### P1：递归暴露存在

至少两个 seed 同时满足：

- Native 累计深度 NLL 的最大值与最小值之差至少 `0.05`；
- 至少两个相邻深度的平均 JS 大于 `1e-4`。

这只说明递归深度改变了词概率场，不说明改变方向必然正确。

### P2：轨迹可复制

用每个 seed 的相邻 NLL 差分向量

```text
delta_d = NLL_d - NLL_(d-1)
```

计算 seed 间 Pearson 相关。三个 seed pair 中至少两个相关系数不低于 `0.50`，
才称为稳定的递归暴露轨迹。

### P3：结构因果

对 `runtime_identity` 与 `pair_break_depth_0` 分别计算其差分向量相对 Native 的
平均绝对偏移。至少一种结构干预在至少两个 seed 中达到 `0.005`，且 source
shuffle 的最终 NLL 高于 Native，才支持轨迹依赖 TreeHeap/source，而不是纯
Decoder 时间步副产物。

## 6. 不预注册单调结论

实验不要求 NLL 随深度单调下降，也不要求每个深度单独不可替代。可能出现：

- 早期层快速缩小候选空间，后期层修正具体 token；
- 某些层增加熵但提高正确候选质量；
- 多层重复写入同一证据，导致后期 NLL 反弹；
- 不同 seed 的轨迹不一致，否定稳定递归协议。

“root 是语言轮廓、leaf 是语言细节”只有在概率轨迹与样本观察共同支持后才能登记，
不能作为本实验的输入假设。

## 7. 后续门

若 P0-P3 通过，才预注册小规模“递归后验更新”训练，将每层 READ 解释为对数概率
证据增量，并与当前 hidden residual READ 做同参数量对照。若 P1 或 P2 失败，停止
该训练方向，先修正 READ 的递归状态定义。

无论结果如何，本实验都不授权 100M-piece Pretrain、产品发布或意识 Claim。

## 8. Evidence

目标目录：

```text
ara/s3-generation/evidence/s3_recursive_depth_probability_exposure_d03/
```

执行代码：

```text
ara/s3-generation/src/s3_recursive_depth_probability_exposure.py
ara/s3-generation/scripts/run_recursive_depth_probability_exposure.sh
```

## 9. 正式结果

正式探针使用 3 个冻结 seed、相同的 256 条 WMT test rows。实现自检与完整深度
原生 READ 对照均通过：最大 logit 绝对误差为 `0.0`，三 seed 的 NLL 误差均小于
`1.4e-7`。P0、P1、P2、P3 全部通过。

三条 Native 相邻深度 NLL 差分轨迹的 Pearson 相关系数为：

```text
seed 10101 vs 10102: 0.9905
seed 10101 vs 10103: 0.9994
seed 10102 vs 10103: 0.9893
```

三个 seed 的均值轨迹如下。`active` 表示该深度仍是真实节点层而非 leaf 饱和复制：

| 深度 | active | NLL | 词表熵 | 正确词概率 | top-1 命中 | 相邻 JS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 367.7266 | 0.0001 | 0.0001 | 0.0001 | - |
| 1 | 1.000 | 259.2811 | 0.0029 | 0.0001 | 0.0001 | 0.0006 |
| 2 | 1.000 | 178.0995 | 0.0149 | 0.0001 | 0.0001 | 0.0043 |
| 3 | 1.000 | 108.8366 | 0.0669 | 0.0001 | 0.0001 | 0.0160 |
| 4 | 1.000 | 62.2547 | 0.2915 | 0.0003 | 0.0003 | 0.0822 |
| 5 | 1.000 | 29.4815 | 1.1880 | 0.0332 | 0.0456 | 0.2642 |
| 6 | 0.886 | 9.0885 | 4.2303 | 0.1366 | 0.2180 | 0.5284 |
| 7 | 0.193 | 5.3003 | 4.9278 | 0.1651 | 0.2630 | 0.1320 |
| 8 | 0.000 | 5.3003 | 4.9278 | 0.1651 | 0.2630 | 0.0000 |

结构干预也改变了完整深度结果。相对 Native，source shuffle 的 NLL 损伤分别为
`+1.7639`、`+1.7304`、`+2.0811`；runtime identity 与第一层 pair break 均明显
改变逐层差分轨迹。因此这不是单纯由 Decoder 时间步产生的固定曲线。

## 10. 结论边界

支持的结论是：当前模型的递归深度以高度可复制、依赖输入与 TreeHeap 结构的方式
逐层改变词表概率场。当前轨迹表现为：root-only 输出极度尖锐但几乎完全错误；继续
递归时，词表熵、正确词概率和 top-1 命中同时上升。因此深层 READ 更像逐层解除错误
坍缩并校正概率场，而不是简单把候选空间越收越窄。

本实验没有证明 root 具有可独立读取的语言轮廓，也没有证明每个深度对应固定语义。
因为 Decoder 只在完整 READ 上训练，截断状态本身属于分布外输入。下一步若要把
“递归后验更新”作为设计，应让训练过程显式看见部分深度，并验证部分深度是否形成
可用而非仅仅可测的概率场。
