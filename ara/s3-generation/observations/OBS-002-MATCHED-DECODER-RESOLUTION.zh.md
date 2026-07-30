# OBS-002：固定 Encoder 下的多 Decoder 分辨率归因观察

状态：观察报告，不注册 Claim

日期：2026-07-30

正式运行：`io` taskd task 69，CUDA，50.6 秒

## 1. 本次要排除什么变量

OBS-001 使用 C08 checkpoint 观察到：开放更深 TreeHeap 状态后，source 相关词
开始进入生成，首 token 概率桶总体变宽。但当时无法区分：

```text
这是 encoder 的信息分布？
还是 C08 EOS-tail decoder 的读取习惯？
```

OBS-002 不修改算法，也不训练新模型。它固定同一个 C04 encoder，让三个与它
兼容的 C06 decoder 读取完全相同的 clean `H_state`：

```text
native_control  ：自由学习 STOP，历史上最终停在 root
leaf_reference  ：强制读取当前最深可见层
depth_floor     ：每个可见深度至少保留 2% route mass
```

这样可以观察：深度状态本身是否携带可用信息，以及不同 READ 方式如何改变它。

## 2. 数据与固定条件

使用 C06 正式实验原有的冻结 held-out WMT split，从中选择 1,000 条：

| Source 长度 | 数量 |
|---|---:|
| 8-12 pieces | 250 |
| 13-16 pieces | 250 |
| 17-24 pieces | 250 |
| 25-32 pieces | 250 |

每个 batch：

1. C04 encoder 只执行一次；
2. 生成 D0-D5 的完整 `levels/masks`；
3. 三个 decoder 读取同一批状态；
4. 不更新任何参数；
5. 分别限制最深可见层；
6. source shuffle 只打乱 batch 中的状态归属，不改变 target；
7. 96 条自由生成按四种长度各取 24 条。

所有 decoder 对应同一 encoder state dictionary digest：

```text
f14529d9f5a8...
```

## 3. 最主要的结果

### 3.1 Native decoder 完全不读取新增深度

| 可见深度 | NLL | 目标平均排名 | Route mass |
|---:|---:|---:|---|
| D0 | 3.5875 | 207.1 | `[1.0]` |
| D1 | 3.5875 | 207.1 | `[1.0, 0]` |
| D2 | 3.5875 | 207.1 | `[1.0, 0, 0]` |
| D3 | 3.5875 | 207.1 | `[1.0, 0, 0, 0]` |
| D4 | 3.5875 | 207.1 | `[1.0, 0, 0, 0, 0]` |
| D5 | 3.5875 | 207.1 | `[1.0, 0, 0, 0, 0, 0]` |

相邻层 JS divergence、decoder context L2 变化均为零。开放深度没有改变 native
decoder 的任何计算结果。

这不是“detail 没有信息”的证据，而是直接证明 native decoder 已把读取协议
固定为 root-only。

### 3.2 Forced-leaf decoder 证明深层状态可以改善任务结果

| 深度 | NLL | 目标平均排名 | Top-1 | 熵 | 相邻层 JS |
|---:|---:|---:|---:|---:|---:|
| D0 | 3.8833 | 232.1 | 0.3617 | 2.980 | 0 |
| D1 | 3.9207 | 238.1 | 0.3575 | 2.983 | 0.00960 |
| D2 | 3.9785 | 248.2 | 0.3463 | 3.199 | 0.09541 |
| D3 | 3.9610 | 238.1 | 0.3374 | 3.511 | 0.08006 |
| D4 | 3.7175 | 210.3 | 0.3735 | 3.620 | 0.05197 |
| D5 | 3.5245 | 192.2 | 0.4069 | 3.413 | 0.05864 |

D0 到 D5：

```text
NLL 改善：0.3588
目标平均排名：232.1 -> 192.2
Top-1：0.3617 -> 0.4069
```

但曲线不是单调的。D1-D3 先恶化，D4-D5 才明显改善。这说明当前各层不是
一条已经校准好的“越深越准确”序列。

按每个样本独立计算：

```text
82.8% 样本的 D5 NLL 优于 D0
平均每样本改善 0.3188
中位改善 0.2513
```

### 3.3 Depth-floor decoder 形成平滑但尚未专业化的累积读取

| 深度 | NLL | 目标平均排名 | Top-1 | 熵 | 相邻 context L2 |
|---:|---:|---:|---:|---:|---:|
| D0 | 3.6495 | 213.3 | 0.3923 | 3.289 | 0 |
| D1 | 3.6474 | 213.1 | 0.3925 | 3.291 | 0.029 |
| D2 | 3.6071 | 208.6 | 0.3962 | 3.321 | 0.286 |
| D3 | 3.5566 | 203.7 | 0.4031 | 3.341 | 0.334 |
| D4 | 3.5105 | 199.3 | 0.4089 | 3.320 | 0.306 |
| D5 | 3.4790 | 195.9 | 0.4136 | 3.259 | 0.299 |

D0 到 D5：

```text
NLL 改善：0.1705
目标平均排名：213.3 -> 195.9
Top-1：0.3923 -> 0.4136
```

每样本统计：

```text
77.1% 样本的 D5 NLL 优于 D0
平均每样本改善 0.1497
中位改善 0.1115
```

完整 D5 的 route 约为：

```text
[0.545, 0.091, 0.091, 0.091, 0.091, 0.091]
```

root 仍占一半以上，其余层几乎均分。当前没有观察到某个深度自然专业化，
但少量、持续的 detail 输入确实改善了输出。

## 4. 长度与深度收益

### 4.1 Depth-floor 的 D0-D5 NLL 改善

| Source 长度 | D0 | D5 | 改善 |
|---|---:|---:|---:|
| 8-12 | 3.3130 | 3.2264 | 0.0866 |
| 13-16 | 3.3331 | 3.1993 | 0.1337 |
| 17-24 | 3.4159 | 3.2436 | 0.1723 |
| 25-32 | 3.9641 | 3.7581 | 0.2060 |

### 4.2 Forced-leaf 的 D0-D5 NLL 改善

| Source 长度 | D0 | D5 | 改善 |
|---|---:|---:|---:|
| 8-12 | 3.5230 | 3.2987 | 0.2243 |
| 13-16 | 3.5234 | 3.2166 | 0.3068 |
| 17-24 | 3.6298 | 3.3053 | 0.3245 |
| 25-32 | 4.2217 | 3.8021 | 0.4196 |

两种递归读取方式都表现出：句子越长，开放完整深度带来的平均收益越大。

每样本的 source 长度与 D0-D5 收益相关系数为：

```text
forced leaf：0.169
depth floor：0.228
```

相关性存在但不强，不能只用长度解释全部收益。

## 5. 为什么 D0 和 D1 几乎相同

64-leaf heap 的 D1 有左右两个半区。但本实验 source 最长为 32 pieces，随后
添加 EOS，只有原长度恰好 32 的少数样本会进入右半区。

1,000 个 root 对应的 D1 有效节点总数只有 `1,025`，意味着绝大多数样本在
D1 只有一个有效 child。此时代数 FOLD/UNFOLD 使该 child 与 root 几乎相同：

```text
D1 parent-child cosine = 0.971
```

所以 D0/D1 相同主要是物理 frame 与输入长度造成的拓扑退化，不应解释成
第一层没有语言意义。

## 6. 状态几何随深度的变化

| 深度 | 有效节点数 | 平均 state norm | 平均维度方差 | Parent-child cosine |
|---:|---:|---:|---:|---:|
| D0 | 1,000 | 3.525 | 0.0364 | - |
| D1 | 1,025 | 3.679 | 0.0413 | 0.971 |
| D2 | 1,585 | 4.597 | 0.0628 | 0.690 |
| D3 | 2,862 | 4.991 | 0.0691 | 0.708 |
| D4 | 5,192 | 5.568 | 0.0893 | 0.730 |
| D5 | 9,851 | 6.614 | 0.1332 | 0.693 |

越深层，state norm 和方差越大。因此逐层输出变化可能同时包含：

- 新增的 source 信息；
- 状态尺度变化；
- decoder 对不同尺度的响应。

目前还不能把所有收益都解释成语义分辨率。需要同 norm 对照才能分离尺度效应。

## 7. Source 因果性

把 batch 中的 TreeHeap 状态滚动给错误 target 后，所有 decoder、所有深度的
NLL 都显著上升。D5 的 source-shuffle damage 为：

```text
native_control：+3.8517
leaf_reference：+3.5173
depth_floor：   +3.9542
```

因此本次逐层收益并非纯 target language prior；被读取的状态与 source identity
存在强因果关系。

## 8. 自由生成观察

96 条生成样例按四个长度组各 24 条。

| Decoder | D0 严重重复率 | D5 严重重复率 | D5 唯一输出比例 |
|---|---:|---:|---:|
| Native | 5.21% | 5.21% | 100% |
| Forced leaf | 7.29% | 1.04% | 100% |
| Depth floor | 4.17% | 2.08% | 98.96% |

深层读取没有引发 C08 那种统一模板坍缩，反而降低了本批样例的严重重复率。
但输出仍有明显术语替换、复制和语法错误，不能当作可用翻译质量证明。

Depth-floor 示例：

```text
Source: Integrated, standards-based certification labeling and reporting
Reference: 集成的、基于标准的认证标签和报告
D0: 整合、报告和标准报告
D5: 依据标签标准、集成和报告
```

```text
Source: Battery Labels - Self-adhesive labels ...
Reference: 电池标签 - 不干胶标签印刷厂家...
D0: - - 印刷印花印刷标签 - 印刷印花...
D5: 印刷 - 印刷印花 - 印刷印花...
```

第一例 D5 增加了“依据/标签/标准”的 source 相关组合；第二例仍然重复，说明
完整深度不是充分条件。

## 9. OBS-001 的“概率桶变宽”是否复现

没有作为跨 decoder 规律复现。

Depth-floor 的平均熵：

```text
D0 3.289
D1 3.291
D2 3.321
D3 3.341
D4 3.320
D5 3.259
```

它先小幅变宽，D5 最终反而略窄于 root。与此同时目标排名和 NLL 持续改善。

Forced-leaf 的熵从 D0 `2.980` 上升到 D4 `3.620`，再在 D5 回落到 `3.413`。

因此 C08 中有效候选数大幅增加，至少部分属于 C08 decoder/EOS-tail 协议，
不能作为当前 encoder 的一般分辨率规律。

## 10. 现在可以陈述的数据事实

1. C04 encoder 的深层状态包含 root-only decoder 没有使用的任务相关信息。
2. Native decoder 的 root-only 行为完全由读取协议造成，不能反推 detail 无用。
3. Forced-leaf 和 depth-floor 都能从 D4-D5 获得稳定平均收益。
4. Depth-floor 提供较平滑的逐层改善；forced-leaf 是非单调跳变。
5. 长句从深层读取获得的收益平均更大。
6. 深层收益具有强 source identity 因果性。
7. 当前 route 没有形成清晰的深度专业化。
8. 深层 state 的 norm/variance 系统性增大，是尚未排除的混杂变量。
9. “深度增加导致 word bag 扩大”不是跨 decoder 的稳定规律。

## 11. 仍然不能得出的结论

本次观察不能证明：

- root 是粗语义、leaf 是细语义；
- 每一层都有独立、可读的语言含义；
- 当前 detail 收益来自内容，而非部分来自 norm 尺度；
- 2% 是最佳 pressure；
- route 应该平均分配；
- TreeHeap 优于匹配的 flat/Transformer；
- 已经形成稳定私有协议。

## 12. 下一次应观察什么

证据将问题缩小到了一个更具体的位置：

> D4-D5 的收益究竟来自新的结构内容，还是来自更大的 state norm 与 decoder
> 对尺度的响应？

下一步不需要训练。可以对同一个 D4/D5 状态做三种匹配干预：

```text
真实 detail
同 norm、跨样本 shuffle 的 detail
同 norm、匹配均值方差的随机 detail
```

同时把各深度 state rescale 到 root 的平均 norm。若 rescale 后真实 detail
仍改善目标排名，而 shuffle/random 不能，就能更明确地把收益归因于内容而非
幅度。完成这个归因后，才适合讨论新的 FOLD 或 READ 算法。

## 13. 数据与复现

- [正式 summary](../evidence/diagnostic_resolution_observation_matched_1k/summary.json)
- [1,000 条逐样本数据](../evidence/diagnostic_resolution_observation_matched_1k/per_example.jsonl)
- [运行说明与 checkpoint 哈希](../evidence/diagnostic_resolution_observation_matched_1k/README.md)
- [观察脚本](../src/s3_resolution_observe_matched.py)
