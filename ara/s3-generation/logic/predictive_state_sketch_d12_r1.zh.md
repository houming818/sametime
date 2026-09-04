# D12-R1：预测状态中程压力阶梯

日期：2026-09-04

Claim：`S3-PREDICTIVE-STATE-SKETCH-D12-R1`

状态：预注册，等待正式运行。

## 1. 为什么继续

D12 的 500-step smoke 已证明固定结果草图的梯度能够到达六个 TreeHeap 层级，且没有
数值故障。梯度校准表明原权重 `0.10` 只产生约为 token CE `0.94%` 的共享梯度压力。
两臂质量几乎重合不能判定该机制无效，只能说明 smoke 的训练长度和压力都不足。

因此本档不扩大模型，也不增加新结构；只把辅助梯度调到约 `5%`，并把训练长度从每臂
500 steps 扩到 10,000 steps，观察学习曲线是否分离。

## 2. 唯一变量

严格配对两臂均从同一个 D10-R1 warm start 开始，使用相同 source checkpoint、预测头
初始化、固定草图、数据顺序、seed、batch、深度轮换和 token CE：

| arm | 预测头训练 | 草图损失进入 TreeHeap |
|---|---:|---:|
| `probe-only` | 是 | 否，detach |
| `predictive-gradient` | 是 | 是 |

辅助损失权重固定为 `0.53`。根据 D12 梯度审计，这对应约 `5%` 的 CE 共享梯度量级。

## 3. 固定预算

```text
GPU: io RTX 3090 24GB
power limit: <= 270.5W
steps: 10,000 per arm, 20,000 total
batch: 16
eval rows: 1,000
generation examples: 64
checkpoint receipt: every 1,000 steps
estimated wall time: 2.5-3.5 hours
```

两臂必须跑满预算。NLL、BLEU、MSE、检索率或重复率的中间非单调变化都不是停止条件。
只有 OOM、CUDA/NVIDIA Xid、NaN/Inf、checkpoint/reload/hash 损坏、功率限制解除，或
step/cursor 不再前进才允许中止。

## 4. 主要观测

1. held-out fixed-sketch MSE 及两臂差值；
2. retrieval@1、shuffle gap、reverse-order gap；
3. raw WMT Test NLL/PPL；
4. 64 条固定生成样本的 BLEU、非空率和相邻重复率；
5. 各层 effective rank、维度标准差和样本间 cosine；
6. 每 1K step 的 step、cursor、token 数与模型/预测头 SHA-256；
7. 最终 checkpoint 重载一致性。

## 5. 判读原则

本档用于判断曲线是否出现可辨识分离，不要求一步完成产品收益。优先看预测 MSE 的相对
改善是否随训练累积，并检查顺序指标是否从负值向零或正值移动；NLL、BLEU和重复率用于
判断代价。即使 10K 时没有收益，也保留完整曲线，不以单点把后续可能的非单调趋势
解释为不存在。

本档不证明私有协议、逻辑或意识，也不自动授权 20% 梯度压力、更多 seed 或更大模型。
