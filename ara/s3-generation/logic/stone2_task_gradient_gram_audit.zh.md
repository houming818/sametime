# STONE-2 C03-D02：真实任务梯度 Gram 审计

日期：2026-08-24

状态：PT 正式审计完成；READ 冲突门通过，branch 冲突门未通过。

## 1. 问题

C03 的 coarse、middle、fine READ 都有正 Shapley 贡献，但两两组合的边际收益为负。
这可能来自三种不同原因：

1. 三组状态包含重复信息，组合后只有边际收益递减；
2. 共享 READ kernel 在不同深度收到方向相反的任务梯度；
3. 共享 branch kernel 在不同深度收到方向相反的路由梯度。

只有第 2 或第 3 项成立，才有根据为 C04 引入分组 kernel 或梯度干预。NLL 的
集合差值本身不能定位梯度冲突。

## 2. 固定对象

- 主 checkpoint：C03 smoke seed 16101 的 `task/PT/checkpoint_best.pt`；
- 数据：checkpoint 自带合同对应的原始 WMT test split；
- 选择：固定 seed，从 source 长度位于 65..128 pieces 的候选中抽取完整 batch，
  每批 8 行；这保证动态 TreeHeap 展开到 128 leaves，完整覆盖深度 0..7；
- 模型、FOLD、Butterfly、Decoder、loss 与 checkpoint 完全冻结；
- 深度组沿用 C03-D01：coarse=`0,1,2`，middle=`3,4,5`，fine=`6,7`。

本实验不更新任何参数，不产生新 checkpoint。

第一次数据可用性审计（taskd 304）发现固定 test split 只有 28 条 source 长度
至少为 65 pieces。taskd 305 进一步发现其中混入了超过 128 pieces、会展开到
256 leaves 的样本；旧代码按 C03-D01 的 128-leaf 分组截断深度，A0 梯度重构门
因此失败。该结果不进入梯度冲突结论。修订后的选择严格限制为 65..128 pieces，
正式 batch 数由不重复候选数量决定，所有门槛保持不变。实现 smoke taskd 302 的
单 batch 只用于验证等函数解绑定和梯度重构。

## 3. 同函数解绑定方法

对训练后的共享 READ kernel 复制八份，对共享 branch kernel 复制七份。每份初始
权重与原共享模块逐字节相同，第 `d` 份只服务深度 `d`。因此解绑定前后前向函数
应当相同：

```text
native shared kernel forward == untied per-depth clone forward
```

在同一个 teacher-forcing token cross-entropy 上反向后，各副本的梯度
`g_d` 是该调用深度对相同参数坐标的任务梯度。若重新把副本视为同一个共享参数，
理论上应满足：

```text
g_shared = sum_d g_d
```

将同一组中的深度梯度求和得到 `g_coarse/g_middle/g_fine`，然后计算：

- 两两 cosine 与 dot product；
- cosine 小于零的 batch 比例；
- `||sum_g|| / sum ||g||`，用于量化共享后的方向抵消；
- 每个深度的梯度 norm；
- 解绑定前后 logits、loss 和共享梯度重构误差。

READ 与 branch 分开报告。depth embedding 每层本来就占不同参数行，不属于共享
参数冲突；recurrent cell 与 output 按生成时间共享，不能在不改变问题定义的情况
下归因到某个树深度，因此不在本次审计中强行拆分。

## 4. 预注册门

- A0：所有量有限；native/untied 最大 logits 差 `<1e-6`；loss 差 `<1e-7`；
  READ 与 branch 的梯度重构相对误差均 `<1e-5`。
- A1 READ 冲突：至少一对深度组的 batch cosine 中位数 `<=-0.05`，且负 cosine
  比例 `>=0.50`。
- A2 branch 冲突：使用与 A1 相同门槛。
- A3 强抵消：某模块的 batch 中位数
  `||sum_g|| / sum ||g|| <=0.80`。

决策：

- A0 失败：实现无效，停止解释；
- A1 通过、A2 不通过：只允许注册分组 READ kernel 的 matched smoke；
- A2 通过、A1 不通过：只允许注册分组 branch kernel 的 matched smoke；
- A1/A2 均通过：允许注册二因素小型 matched smoke，不允许直接长训；
- A1/A2 均失败：不拆 kernel。C03-D01 的负交互优先解释为冗余、饱和或其他
  非线性链路，继续定位 state 与 query 增益，而不是修改参数共享方式。

A3 只衡量抵消强度，不单独授权架构变更。

## 5. 边界

该审计可以定位共享 READ/branch 参数上的优化方向冲突，但不能证明某个深度具有
人类可读语义，也不能证明分组参数一定改善 NLL。它不补签 C03 的 S7，不授权
100M-piece 正式训练。

## 6. 执行结果

- 实现 smoke：io taskd `302`；
- 首次正式运行：taskd `303`，因混入短树导致空 fine branch，失败；
- 数据计数：taskd `304/306`；
- 混宽运行：taskd `305`，发现 256-leaf 样本被错误截断，A0 按预注册停止；
- 有效正式运行：taskd `307`，固定 65..128 pieces，3 batch × 8 行。

有效运行中，native 与解绑定模型的前向完全匹配，READ/branch 梯度重构最大相对
误差分别为 `2.24e-7` 与 `3.96e-7`。A0 通过。

READ 的组级结果：

| 组对 | cosine 中位数 | 负 cosine 比例 |
|---|---:|---:|
| coarse : middle | -0.1596 | 0.6667 |
| coarse : fine | 0.0534 | 0.3333 |
| middle : fine | 0.0973 | 0.3333 |

READ 的 `||sum_g|| / sum||g||` 中位数为 `0.7003`。A1 与 A3 通过。branch 的
三组 cosine 中位数均为正，抵消比中位数为 `0.8202`，A2 未通过。

结论限于当前 PT checkpoint：共享 READ kernel 的 coarse/middle 调用方向存在
可复现任务梯度冲突；共享 branch 不是本次定位到的主要冲突点。按预注册决策，
下一步只允许注册分组 READ kernel 的 matched smoke。不能据此宣称分组 kernel
已经改善任务质量，也不能修改 branch、FOLD 或数据合同。
