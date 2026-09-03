# D11：TreeHeap 协议容量第一档尺度阶梯

日期：2026-09-02

Claim：`S3-STRUCTURAL-PROTOCOL-CAPACITY-LADDER-D11`

状态：已预注册，等待 D10-R1 完成后自动启动。

## 1. 问题

D10 回答清洗全集能否让当前 TreeHeap 协议继续学习。D11 不再增加语料或更改损失，而是
回答另一个问题：在相同 D10 checkpoint、相同样本顺序、相同训练步数下，增加 TreeHeap
协议容量是否带来超过“再训练一遍”的收益。

不能把 D10 与更大模型直接比较，因为更大模型还会额外看到一遍数据。因此第一档必须有
两个配对 arm：

| 名称 | 含义 |
|---|---|
| `TreeHeap-63M` | 完整 D10 最佳 checkpoint，继续训练 25K steps |
| `TreeHeap-106M` | 同一 checkpoint 加一个 384 维残差 TreeHeap 通道，再训练相同步数 |

名称使用实际运行时总参数量的近似值；正式报告以程序记录的精确参数数为准。

## 2. 为什么采用残差 TreeHeap 通道

直接把 256 维状态改成 384 维会改变 LayerNorm、GRU 和词表矩阵，无法保证扩容模型在
step 0 与原模型表示同一个函数。那样如果结果变化，我们无法区分是容量作用还是初始化
破坏。

新增通道独立执行：

```text
冻结的 source TreeHeap
-> 384 维投影
-> adjacent-subheap ownership READ
-> 递归 FOLD
-> 多层 READ + K_up
-> residual logits
```

最终 logits 为：

```text
logits = logits_D10 + tanh(g) * logits_extra
```

其中 `g=0`。因此扩容模型在训练前与 D10 的输出严格相同；第一步先学习是否打开新增
通道，之后梯度才逐渐写入该通道。新增容量不是 flat 输出适配器，而是一条完整的
TreeHeap 编码、递归 FOLD 和多层解码路径。

## 3. 固定合同

```text
source: C12 formal seed10101 READ，冻结
warm start: D10-R1 完成后的 best checkpoint
data: NioClean-ZHEN-S098-7M-v2
external eval: 原始 WMT 固定 valid/test
seed: 11101
ownership seed: 11102
batch: 16
steps: 25000 each arm
learning rate: 1e-4
depth exposure: 5 / 6 / 7
loss: target token cross entropy
GPU: io RTX 3090, power limit <= 270.5 W
```

两臂从语料 cursor 0 开始，读取完全相同的 25K batches。D10 的完整一遍数据暴露属于共同
初始化，不计入本阶的 arm 间差异。

## 4. 启航门

任务在 `nio-taskd` 中排在 D10-R1 后面。启动时必须确认：

1. D10 已遍历完整 release；
2. D10 的输入因果、结构、冻结底座和 reload 门通过；
3. 270W 功率限制仍生效；
4. 真实 checkpoint smoke 中两臂 step-0 NLL 差不超过 `1e-8`；
5. 两臂 smoke 均有有限梯度、结构覆盖、checkpoint 重载和非空生成。

任一门失败，正式 25K 配对训练不会启动。

## 5. 第一档预测

扩容只在以下条件全部满足时得到支持：

```text
NLL(TreeHeap-63M) - NLL(TreeHeap-106M) >= 0.03
BLEU4(TreeHeap-106M) - BLEU4(TreeHeap-63M) >= -0.50
repetition(TreeHeap-106M) - repetition(TreeHeap-63M) <= 0.02
```

同时要求同 source、同 warm start、同数据、同 seed、同 cursor、同 steps，且扩容臂继续
通过输入因果和结构门。

NLL 是容量收益的主指标；BLEU 和重复率是防止模型用更低 teacher-forced loss 换取更差
自由生成的保护门。单 seed 通过只授权设计相邻的约 150M 档，不授权直接跳到 300M，也
不构成产品质量结论。

## 6. 停止规则

- D10 交接门失败：不启动；
- smoke 失败、OOM、CUDA/Xid、NaN/Inf、哈希或 reload 失败：立即停止；
- 106M 未达到 `0.03` NLL 配对收益：停止自动扩容；
- 生成保护门失败：停止自动扩容并检查容量是否只改善 teacher forcing；
- 第一档通过：只预注册下一档，不在同一任务中无界追加模型。

Evidence：

```text
ara/s3-generation/evidence/s3_structural_protocol_capacity_ladder_d11/
```
