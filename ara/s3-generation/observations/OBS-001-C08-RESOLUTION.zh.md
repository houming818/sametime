# OBS-001：C08 Checkpoint 的 TreeHeap 分辨率切片观察

状态：观察报告，不注册 Claim

日期：2026-07-30

作者：Codex / Nio Log Squad Review Engineer

原始运行：`io` taskd task 63，CUDA，4.5 秒

## 1. 为什么做这次观察

当前研究暂时退出 STONE 产品里程碑，不继续根据 NLL、BLEU 或 CLI 表现直接
修改模型。我们先观察已有 TreeHeap checkpoint 内部发生了什么，再根据重复
出现的数据规律设计算法。

本次只回答一个很小的问题：

> 对同一个已经编码完成的 `H_state`，让同一个 decoder 依次看到 root、
> root+depth1，直到完整可见层时，概率桶和生成文本实际怎样变化？

本报告不预设：

- root 是句意轮廓；
- leaf 是语言细节；
- 深度越大概率越集中；
- 当前现象是 TreeHeap 的普遍规律。

## 2. 为什么使用 C08

C08 不是最新或最好的模型，但它具有这次切片需要的完整实验条件：

1. `io` 上存在匹配的 encoder、decoder 和 tokenizer；
2. encoder 会产生真实的多层 `H_state`；
3. `CanonicalTreeHeap.visible()` 原生支持限制最大可见层；
4. C08 decoder 使用 C06 depth floor，所有深度都曾参与训练；
5. 已发布 checkpoint 可复查，不需要临时混合不同实验参数。

使用的文件：

```text
encoder-growth-step62500.pt
decoder-eos-tail.pt
sp-bpe-massive.model
```

C08 同时带有固定 64-leaf frame、EOS tail、冻结 C04 encoder 和 2% depth
floor。这些设计会影响结果，因此本报告只描述 C08，不代表 TreeHeap 一般性质。

## 3. 实验装置

### 3.1 固定变量

对每个输入：

1. 只调用一次 encoder；
2. 得到一份完整 `H_state`；
3. 所有深度切片复用这份状态；
4. decoder 参数、greedy 规则、BOS/EOS 和最大生成长度保持不变；
5. 不训练、不更新任何参数。

唯一变化是：

```text
D0 = root
D1 = root + depth1
D2 = root + depth1 + depth2
...
D5 = root + ... + depth5
```

每个输入都保存了 root SHA-256，用于确认切片来自同一次编码状态。

### 3.2 输入场景

```text
The earth is round.
The apple is sweet.
Why is the window wet?
A cat is eating some food.
I arrived home at seven o'clock.
```

这五句话只用于第一张显微镜切片，不构成统计充分的语言样本。

### 3.3 记录项目

每个深度记录：

- 首 token 概率熵；
- 有效候选数 `exp(entropy)`；
- Top-10 总概率；
- Top-10 token；
- route mass；
- greedy 生成文本；
- 生成 token 数。

## 4. 总体概率桶变化

| 最深可见层 | 平均熵（nats） | 平均有效候选数 | 平均 Top-10 概率质量 |
|---:|---:|---:|---:|
| D0 | 5.646 | 311.1 | 0.407 |
| D1 | 5.850 | 365.6 | 0.364 |
| D2 | 5.905 | 384.6 | 0.354 |
| D3 | 5.999 | 420.0 | 0.338 |
| D4 | 6.120 | 473.4 | 0.316 |
| D5 | 6.014 | 439.2 | 0.304 |

在这个 checkpoint 上，从 D0 到 D4：

- 有效候选数从 `311.1` 增加到 `473.4`，增加约 52%；
- Top-10 概率质量从 `0.407` 降到 `0.316`；
- 概率没有逐层变尖，而是总体变宽。

五个输入共有 25 次相邻深度变化，其中：

```text
熵上升：20 次
熵下降： 5 次
```

因此，“开放深层后概率桶总体扩大”是本次数据的直接描述。它为什么扩大，
目前还不能确定。

## 5. Top-10 候选集合变化

相邻深度 Top-10 集合的平均 Jaccard 相似度：

| 深度变化 | 平均 Top-10 Jaccard |
|---|---:|
| D0 -> D1 | 0.456 |
| D1 -> D2 | 0.758 |
| D2 -> D3 | 0.788 |
| D3 -> D4 | 0.758 |
| D4 -> D5 | 0.352 |

观察到两个变化较大的边界：

- `D0 -> D1`：root-only 候选与加入第一层后的候选明显换组；
- `D4 -> D5`：加入最深可见 detail 时，Top-10 再次明显换组。

D1 到 D4 的候选集合相对稳定，主要变化可能发生在候选概率重分配，而不是
每层都完全换一套词。由于本次没有保存完整词表分布，还不能计算严格的层间
JS/KL divergence。

## 6. 五个场景的逐层生成

### 6.1 `The earth is round.`

| 层 | 生成观察 |
|---|---|
| D0 | 无关且包含无效 piece 的片段 |
| D1 | 多个无效 piece 和“子” |
| D2 | 无效 piece，随后出现“地道” |
| D3 | `这是一间大地的。` |
| D4 | `圆圆的圆珠轮在地上。` |
| D5 | `圆圆圆珠在地上。` |

“地、圆”到 D3-D5 才进入生成，但句意和语法仍然错误。

### 6.2 `The apple is sweet.`

| 层 | 生成观察 |
|---|---|
| D0 | 无关的“破碎机”片段 |
| D1 | 重复无关片段 |
| D2 | 无效 piece 和标点 |
| D3 | `这是一种很高的。` |
| D4 | `是苹果的。` |
| D5 | `甜菜,这是一种甜美的甜蜜。` |

“苹果、甜”在 D4-D5 出现，但 D5 同时发生语义漂移和词语重复。

### 6.3 `Why is the window wet?`

| 层 | 生成观察 |
|---|---|
| D0 | 无关片段 |
| D1 | 无关的“破碎机/子”片段 |
| D2 | 仍以无关片段为主 |
| D3 | `为什么?` 开始出现 |
| D4 | `窗口, 为什么?` |
| D5 | `窗子为什么会不当?` |

疑问结构和“窗口/窗子”在深层出现；“wet”的对应信息没有正确生成。

### 6.4 `A cat is eating some food.`

| 层 | 生成观察 |
|---|---|
| D0 | 无关片段 |
| D1 | 无效 piece 为主 |
| D2 | 无效 piece 为主 |
| D3 | `这是一些时候,就吃了。` |
| D4 | `吃过早餐,吃惊吃。` |
| D5 | `吃过早餐,吃惊吃。` |

“吃”在 D3 后出现，但“cat”和完整事件关系没有恢复。

### 6.5 `I arrived home at seven o'clock.`

| 层 | 生成观察 |
|---|---|
| D0 | 无关片段 |
| D1 | 开始出现“我们/了”，仍有大量无效 piece |
| D2 | 出现“我”，仍不可用 |
| D3 | `我真是...`，与原句关系较弱 |
| D4 | `回家,我回家了。` |
| D5 | `我回家了,我回家了。` |

“我、回家”在 D4-D5 形成明显 source 关联；七点信息没有恢复，D5 出现重复。

## 7. Route 的直接观察

截断到 D0 时，root 被迫承担 100% route mass。随着可见层增加，route mass
大致分散到各层。例如完整 D5 常见近似值：

```text
[0.18, 0.18, 0.15, 0.15, 0.15, 0.18]
```

目前没有看到某一深度获得压倒性的 learned route 权重。由于 C08 使用固定
depth floor，而且截断会重新归一化读取层，不能把接近平均分配解释成自然形成
的分辨率协议。

## 8. 可以直接陈述的事实

1. 同一 `H_state` 在不同可见深度下产生了不同概率桶和不同文本。
2. D0 root-only 没有生成可识别的粗略句意。
3. source 相关词通常到 D3-D5 才明显进入生成。
4. 增加可见深度时，首 token 分布总体变宽，而不是总体变尖。
5. D1-D4 的 Top-10 集合较稳定，D0/D1 和 D4/D5 是候选换组较大的边界。
6. 深层虽然增加 source 相关信息，也同时带来重复、漂移和错误组合。
7. 当前 route 没有表现出明确的深度专业化。

## 9. 现在不能得出的结论

本次数据不能说明：

- TreeHeap 的高层天然表示轮廓；
- 深层天然表示语言细节；
- 更宽的概率桶一定包含更多有效信息；
- C08 的现象来自 encoder，而不是 decoder；
- TreeHeap 已形成稳定私有协议；
- 增加深度可以单调提高翻译质量；
- 这五句话代表真实语料总体规律。

特别需要注意：root-only 和其他截断状态并不是 C08 的独立训练任务。它们是
对一个全层训练 decoder 的干预结果。

## 10. 本次观察暴露的测量缺口

下一轮若继续观察，应补采以下数据，而不是立即修改算法：

1. 保存每个深度的完整 logits，以计算层间 JS/KL divergence；
2. 保存 decoder context vector，区分状态变化和 output head 放大；
3. 保存每个生成 step，而不只保存首 token 和最终文本；
4. 使用带 reference 的 held-out WMT，记录目标 token rank 和逐层 NLL；
5. 记录各层 node norm、variance、effective rank 和 parent-child distance；
6. 用同一个 C04 encoder 比较 C04、C06、C08 decoder；
7. 增加长度分层和至少 1,000 条样本；
8. 分开记录乱码、重复、source 词出现和正确关系恢复。

## 11. 当前最谨慎的项目判断

这次观察没有发现“root 是轮廓、深层逐步补细节”的完整现象。

它发现的是一个更原始的过程：

```text
root-only：缺少足够条件信息，输出受无关高频模板支配
逐层开放：source 相关词逐渐进入概率桶和生成
完整深度：相关性增加，但协议仍不稳定，出现重复和错误组合
```

这个过程究竟来自 encoder 的信息分布、decoder 的读取习惯、depth floor，
还是 EOS-tail 训练协议，需要通过 matched decoder 观察来区分。现在不应据此
改 FOLD、增加 split，或建立新的产品里程碑。

## 12. 数据与复现

- [原始 JSON](../evidence/diagnostic_resolution_observation_c08/result.json)
- [运行说明](../evidence/diagnostic_resolution_observation_c08/README.md)
- [诊断脚本](../src/s3_resolution_observe_existing.py)
