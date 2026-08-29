# D08：TreeHeap 协议槽位的结构覆盖

状态：D08R1 三 seed 中型验证正式支持（结构机制）；不代表产品生成质量通过。

Claim：`S3-STRUCTURAL-SLOT-OWNERSHIP-D08`

## 1. D07R2 留下的问题

D07R2 使用递归深度控制有限协议槽位，并用共享有界增益把协议贡献从负值修正为一致的
弱正值。打乱槽位使 test NLL 增加 `0.096--0.142`，清空槽位使 NLL 增加
`0.036--0.071`。后者没有达到预注册 `+0.10`，不同深度的生成也基本相同。

当前日志还显示，部分递归分支熵接近零。它证明路由变得高度确定，但没有直接证明不同
槽位是否选择了同一路径。D08 增加显式的槽位路径重叠和终点覆盖测量，并测试一个最小
结构修正：槽位只负责各自的 source subheap。

## 2. Claim

若 D07R2 的主要瓶颈是自由 query 重复读取少数容易降低损失的区域，那么给槽位分配互不
重叠的递归地址责任，应当：

1. 降低实际终点路由重叠；
2. 提高不同槽位 argmax 终点的覆盖；
3. 增强 native 协议相对 shuffle 和 zero 的 NLL 优势；
4. 使新增深度/容量提供互补信息，而不是复制同一信号。

结构先验只规定“去哪个 subheap 读取”。共享 READ kernel 和完整目标交叉熵仍然决定
“在那里保存什么”。不提供主语、谓语、宾语、摘要或目标前缀标签。

## 3. 三个同初始化 Arm

| Arm | 地址规则 | 目的 |
|---|---|---|
| `free` | 每个 query 可读取整棵 source TreeHeap | 复现 D07R2 自由竞争基线 |
| `subheap` | 有效 source leaf 按递归邻接区间分成互不重叠的责任域 | 测试 TreeHeap 结构覆盖 |
| `random` | 每槽位分配同样数量但随机打散的 leaf 地址 | 区分结构邻接与单纯强制分工 |

对 `subheap`，一个槽位在浅层可以与其他槽位共享祖先；一旦树地址能够区分责任域，它只
能沿覆盖自己 leaf 区间的子节点继续 READ。对 `random`，容量和不重叠性相同，但相邻
leaf 被打散。

三组保持相同：

```text
source checkpoint
训练/验证/测试行及其 SHA-256
参数初始化 seed
batch 顺序和 depth 顺序
steps、学习率和梯度裁剪
冻结 source TreeHeap 与语言骨架
共享有界 protocol gain
完整目标 token 交叉熵
```

## 4. 新诊断

每个 batch 在最终 source 深度记录：

```text
route_pair_overlap = mean_ij sum_n min(p_i(n), p_j(n))
route_pair_cosine  = mean_ij cosine(p_i, p_j)
argmax_coverage    = unique(argmax_i p_i) / active_slots
owner_leaf_coverage = union(owner_i) / valid_source_leaves
```

相同路由的 overlap 接近 1，互不重叠路由接近 0。`argmax_coverage=1` 表示每个活动槽位
最终选择了不同终点。ownership 指标只验证地址约束被正确执行，不作为语义成功证据。

## 5. Smoke 配置

```text
seed: 10801
arms: free / subheap / random
steps per arm: 600
train / valid / test rows: 2048 / 128 / 128
depths: 5 / 6 / 7
max protocol slots: 32
device: single RTX 3090
```

预计单班 8--15 分钟。任何 OOM、CUDA 错误、NaN/Inf、冻结哈希变化或 evidence 不完整
都立即停止。

## 6. Predict 与门槛

### P0：实现合同

- 三组初始化、数据与训练 schedule 哈希一致；
- source 与语言骨架训练前后哈希不变；
- target 不进入 compressor，target 长度不进入容量；
- 不存在 Transformer、自注意力或 flat `L x L` 路由表。

### P1：结构约束生效

- `subheap` 与 `random` 的 owner leaf coverage 均为 `1.0`；
- 两组最终 route pair overlap 均小于 `0.05`；
- `subheap` 的 argmax coverage 至少为 `0.90`。

### P2：TreeHeap 邻接优于自由重复和随机分工

至少两个深度满足：

```text
NLL(subheap) <= NLL(free)   - 0.02
NLL(subheap) <= NLL(random) - 0.02
```

若 subheap 与 random 相同，只能支持“强制分工有用”，不能支持递归邻接有用。

### P3：输入协议达到因果门

`subheap` 至少两个深度同时满足：

```text
NLL(shuffle) - NLL(native) >= 0.10
NLL(zero)    - NLL(native) >= 0.10
```

### P4：容量方向

```text
NLL(depth=7) <= NLL(depth=6) + 0.05
NLL(depth=6) <= NLL(depth=5) + 0.05
```

### P5：训练与生成没有数值坍缩

两端 READ、slot kernel 和 gain 均有非零有限梯度；槽位方差大于 `1e-4`；记录固定生成、
BLEU 和重复率，但 smoke 不用生成质量单独升级 Claim。

## 7. 决策

- P0/P1 失败：实现无效，停止；
- P2 失败：结构 ownership 没有改善当前瓶颈，停止该候选；
- P2 通过、P3 失败：分工改善了重建，但私有协议仍太弱，只允许修改接口后重试；
- P0--P5 全通过：允许排 3 seed 中等训练，不直接进入产品长训。

Evidence：

```text
ara/s3-generation/evidence/s3_structural_slot_ownership_d08/
```

## 8. D08 smoke 结果与审计

seed 10801 的三个 arm 均完成 600 steps，初始化、数据、source 哈希和冻结语言骨架一致，
六个预注册门均通过。subheap 相对两个对照的 Test NLL 改善如下：

| depth | 相对 free | 相对 random | shuffle delta | zero delta |
|---:|---:|---:|---:|---:|
| 5 | 0.1412 | 0.0302 | 0.3297 | 0.2238 |
| 6 | 0.2112 | 0.0889 | 0.4127 | 0.2919 |
| 7 | 0.2027 | 0.0894 | 0.3594 | 0.2782 |

free 的最终 route overlap 为 `0.956--0.981`，subheap 为 `0`；subheap 的 argmax
coverage 与 owner coverage 均为 `1.0`。这支持“自由 query 会重复读取，而结构责任域能增强
输入协议”的单 seed 证据。

复核实现时发现：D08 的 random 分组 seed 含 batch 内行号。同一个样本换到 batch 的另一
位置时，随机责任域也会改变。因此 subheap 与 random 的差距可能同时包含：

1. 邻接子堆相对离散地址的差异；
2. 固定协议相对随 batch 位置漂移协议的差异。

这不影响 subheap 对 free 的结果，但会污染“递归邻接优于随机分工”的解释。故不直接进入
多 seed 正式训练。

## 9. D08R1 复验合同

D08R1 只做两个预注册修正，其他配置和门槛保持不变：

1. random 分组仅由 `seed + valid source length + budget` 决定，不再依赖 batch 位置；
2. 除全体数值方差外，增加真正的槽间方差：对每个样本先沿 slot 维计算方差，再对隐维和
   样本求均值；要求三个深度都大于 `1e-8`。

先复跑 seed 10801 的 600-step smoke。若 P0--P5 再次全部通过，才排 seeds
`10811/10812/10813` 的中型验证；若 P2 不再通过，则 D08 只能支持“固定分工有用”，不能
支持“递归邻接有用”。

## 10. D08R1 smoke 结果

D08R1 再次通过 P0--P5。subheap 相对 free 的 depth 5/6/7 NLL 优势为
`0.1412/0.2112/0.2027`，相对稳定 random 为 `0.0280/0.0664/0.0537`。后两个
深度的 random 差距虽比 D08 略小，但仍超过预注册 `0.02`，因此 batch-position 混杂不是
主要收益来源。允许进入中型复验。

## 11. 三 seed 中型验证

```text
seeds: 10811 / 10812 / 10813
arms per seed: free / subheap / stable-random
steps per arm: 3000
train / valid / test rows: 20000 / 512 / 512
batch: 8
depths: 5 / 6 / 7
device: single RTX 3090, serial queue, power limit <= 270.5 W
```

正式支持需要：

1. 三个 seed 的 P0/P1/P4/P5 全通过；
2. 至少两个 seed 分别通过 P2 和 P3；
3. 至少两个深度的三 seed 中位数同时满足 `vs_free >= 0.02`、`vs_random >= 0.02`；
4. 至少两个深度的 shuffle/zero delta 中位数均不小于 `0.10`。

任一 seed 出现 OOM、CUDA 错误、NaN/Inf、冻结哈希变化或 evidence 损坏，停止后续队列。
本阶段仍只验证结构协议，不以 BLEU 单独升级为产品训练。

## 12. 三 seed 正式结果

任务 333 完整运行 `6347` 秒，seeds `10811/10812/10813` 均通过 P0--P5，远程与
本地独立聚合结论均为 `formal_supported`。

| depth | vs free 中位数 | vs stable-random 中位数 | shuffle delta 中位数 | zero delta 中位数 | BLEU4 中位数 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.0327 | 0.0619 | 1.1906 | 0.5828 | 3.6594 |
| 6 | 0.0503 | 0.0632 | 1.1642 | 0.6232 | 2.7888 |
| 7 | 0.0320 | 0.0592 | 0.9898 | 0.6068 | 3.6300 |

正式门的复算结果：

```text
implementation_all: true
quality_seed_passes: 3 / 3
causality_seed_passes: 3 / 3
median_quality_depths: 3 / 3
median_causality_depths: 3 / 3
```

因此当前证据支持：在相同初始化、数据、训练步数与参数量下，把协议槽位绑定到互不重叠
的相邻递归 source subheap，比允许所有槽自由读取、或绑定到同容量但离散的随机 leaf
集合，更稳定地降低 Test NLL；打乱或清空 subheap 协议会产生远高于门槛的损失。这是
TreeHeap 递归邻接参与私有协议的三 seed 因果证据。

边界同样明确：BLEU4 仍低，尚未证明可用翻译、对话或长文本生成。本结果允许把
`subheap ownership` 纳入下一版候选架构，但产品训练仍需独立 claim、数据规模和生成门。
