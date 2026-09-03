# D12：TreeHeap 预测状态随机草图

日期：2026-09-03

Claim：`S3-PREDICTIVE-STATE-SKETCH-D12`

状态：GPU 配对 smoke 完成；梯度机制成立，预测质量收益仍未确定。

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

## 9. Smoke 结果

`io.grepcode.cn` 上的 taskd 任务 `356` 完成两臂各 500 steps，每臂处理 190,541 个
目标 token，总墙钟时间 491 秒。RTX 3090 功率上限保持 270W，没有 OOM、NaN/Inf、
checkpoint 或重载异常。

两臂 step-0 NLL 和预测 MSE 完全相同。`probe-only` 的六层辅助梯度均严格为零；
`predictive-gradient` 的六层梯度范数为：

```text
[0.000782, 0.000825, 0.000854, 0.000855, 0.000873, 0.002181]
```

这支持最窄的机制结论：固定预测草图损失可以穿过现有 Decoder 卷积，向六个 TreeHeap
层级发送有限、非零的梯度。

| 指标 | probe-only | predictive-gradient | 差异 |
|---|---:|---:|---:|
| held-out NLL | 4.180471 | 4.180467 | predictive 改善 0.000004 |
| held-out sketch MSE | 1.040171 | 1.040140 | predictive 改善 0.000031 |
| retrieval@1 | 0.101562 | 0.101562 | 0 |
| shuffle gap | 0.005892 | 0.005892 | 近似 0 |
| reverse-order gap | -0.002693 | -0.002677 | 仍为负 |
| median token BLEU-4（16例） | 10.749587 | 11.764335 | +1.014748，探索性 |
| 最大相邻重复率 | 0.032581 | 0.031785 | -0.000797 |

预测头在 `probe-only` 中也把 MSE 从 `1.048849` 降到 `1.040171`。开放 TreeHeap 梯度后
额外获得的 `0.000031` 只占约十万分之三，远不足以声称表示质量已经得到有意义改善。
本次预注册的 `MSE > 0` 存在性门确实通过，但它不能区分可复现收益与微小数值差异。

正 shuffle gap 表示原有 TreeHeap 状态已经包含一部分可由线性头读取的样本相关目标信息；
负 reverse-order gap 则表示当前头尚未学会偏好真实顺序。BLEU 只基于 16 个生成样本，
而且 NLL 几乎不变，因此暂不把约 1 点的差异解释为产品收益。

六层 effective rank 均约为 `71-76`，没有观察到表示坍缩；两臂几何量的变化很小。

## 10. 下一步

不直接启动长训练。先增加只读梯度校准，分别计算 token CE 与 predictive loss 到相同
TreeHeap 层级和参数组的梯度范数、夹角。随后预注册一个小型比例阶梯，使辅助梯度约占
CE 共享梯度的 `1% / 5% / 20%`，而不是盲猜 loss 权重。评价至少扩大到 1,000 条，保留
固定训练预算，不因中间质量曲线非单调而停止。

任务 `358` 已使用完成 500 steps 的 `probe-only` checkpoint 对后续 8 个 batch 做只读
校准。未乘 loss weight 时，预测梯度相对 CE 的整棵协议树梯度比例为：

```text
min       0.057470
median    0.094421
max       0.121317
```

两类梯度在协议树上的平均 cosine 为 `0.008896`，整体接近正交。compressor 参数组上的
平均 cosine 为 `-0.053201`，存在很弱的局部冲突；`up_kernel` 为 `0.005643`。没有证据
表明当前主要问题是强烈对冲。

原 smoke 使用 `predictive_weight=0.10`，所以预测梯度实际约为 CE 梯度的 `0.94%`，这能
解释两臂结果几乎重合。根据中位比例，目标压力对应的校准权重为：

| 目标预测/CE 梯度比 | 建议 loss weight |
|---:|---:|
| 1% | 0.1059 |
| 5% | 0.5295 |
| 20% | 2.1182 |

因此下一档应先测相邻的 `0.53`，不直接跳到 `2.12`。这些值只负责确定梯度量级，不是
已经授权的长训练超参数。

正式 evidence：`../evidence/s3_predictive_state_sketch_d12/smoke_seed11201/`。
