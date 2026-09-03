# D12：TreeHeap 预测状态随机草图

日期：2026-09-03

Claim：`S3-PREDICTIVE-STATE-SKETCH-D12`

状态：预注册，等待 CPU 合同和 GPU 配对 smoke。

## 1. 问题

D11-R1 表明，在相同 25K 训练预算下，从 63M 增加到 106M 只改善 `0.004722`
Test NLL。容量可以训练且没有破坏系统，但增加的参数没有明显转化为任务质量。因此 D12
不继续扩大模型，而是改变 TreeHeap 中间状态收到的学习信号。

当前 token CE 只直接评价 Decoder 最终输出。D12 检验一个较窄的机制问题：如果每个
卷积后 TreeHeap 层级都被要求预测真实目标序列的固定、顺序敏感随机草图，梯度是否能
到达这些层，并使其在 held-out 数据上更好地表示目标结果，同时不破坏 token NLL。

## 2. 固定目标

为每个 token 生成不可训练的随机符号码 `R(y)`，为每个目标位置生成不可训练的位置符号
码 `Q(k)`。目标序列的草图为：

```text
phi(y_1...y_T) = sum_k R(y_k) * Q(k) / sqrt(T)
```

这里 `*` 是逐坐标符号绑定。token 和位置表都作为 buffer 保存，不参与优化，并记录
SHA-256。交换 token 顺序会改变草图；PAD 不参与计算。草图宽度固定为 128。

这不是外部模型 embedding，也不引入教师语义。它是实际目标序列的固定有限维观测。
在条件分布意义下，每条训练记录提供一次未来结果样本；MSE 学习的是该固定特征的条件
均值近似。

## 3. TreeHeap 接口

协议 slots 先按现有 `fold_protocol` 形成 TreeHeap，再经过现有 Decoder `up_kernel`
卷积。对卷积后的每个层级，按有效节点做 `sum/sqrt(count)` 并执行无参数 LayerNorm，得到
层级联合状态。六个独立线性头分别把六层状态映射到同一个 128 维固定结果空间。

第一步监督的是“每层联合状态”，不是声称每个 parent 已经独立获得语言含义。这样可以
先证明梯度通道和预测状态目标有效，再决定是否把目标细分到单节点。

## 4. 严格配对实验

两臂使用完全相同的 D10-R1 checkpoint、预测头、固定草图、数据顺序、seed、batch 和
500 steps：

| arm | token CE | 预测头自身训练 | 预测损失到 TreeHeap 的梯度 |
|---|---|---|---|
| `probe-only` | 开启 | 开启 | `detach`，关闭 |
| `predictive-gradient` | 开启 | 开启 | 开启 |

因此，两臂都允许预测头学习。唯一控制变量是预测损失能否改变 TreeHeap 状态。辅助损失
权重为 `0.10`，不替换原 token CE。

## 5. 数据与平台

```text
source checkpoint: C12 multi-level READ best
warm start: D10-R1 completed best checkpoint
task data: NioClean-ZHEN-S098-7M-v2
evaluation: held-out raw WMT partition
batch: 16
steps: 500 per arm
TreeHeap depths: 5, 6, 7 rotating
GPU: io RTX 3090 24GB, power limit <= 270.5W
```

## 6. 观测

除 NLL、PPL、BLEU、非空率和重复率外，记录：

1. 每个卷积层收到的纯辅助损失梯度范数；
2. held-out fixed-sketch MSE；
3. batch 内目标检索 `retrieval@1`；
4. 正确目标相对 batch shuffle 目标的 MSE gap；
5. 正确顺序相对 reverse 顺序的 MSE gap；
6. 每层 effective rank、维度标准差和样本间平均 cosine；
7. 固定草图、模型和预测头的 SHA-256；
8. checkpoint 重载一致性。

## 7. 机制门

本次 smoke 支持机制需要同时满足：

1. 两臂 step-0 NLL 与预测 MSE 差不超过 `1e-9`；
2. 固定草图哈希相同，源模型保持冻结；
3. `probe-only` 的辅助梯度不能到达 TreeHeap；
4. `predictive-gradient` 至少让六层中的四层收到大于 `1e-8` 的辅助梯度；
5. predictive arm 的 held-out sketch MSE 优于 probe arm；
6. predictive arm 相对 probe arm 的 token NLL 恶化不超过 `0.10`；
7. 两臂数值有限、完成全部步数并通过重载。

BLEU、重复率、检索率和顺序 gap 在本次短 smoke 中作为观察，不作为中途停止门。无论
结果如何都不自动进入长训练。

## 8. Claim 边界

即使通过，也只说明固定预测状态目标能够形成一条可训练、可泛化检查的 TreeHeap 层级
梯度通道。它不证明已经涌现私有协议、逻辑或意识，也不证明长训练 BLEU 必然提升。
