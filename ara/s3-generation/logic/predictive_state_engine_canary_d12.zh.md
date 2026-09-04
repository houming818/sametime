# D12-E1：消费级 GPU 吞吐引擎 Canary

日期：2026-09-04

Claim：`S3-PREDICTIVE-STATE-ENGINE-CANARY-D12-E1`

状态：预注册，排在 D12-R1 之后运行。

## 1. 目的

Prometheus 最近七天显示，`io` 的 RTX 3090 平均利用率只有 `17.2%`；即使排除利用率
低于 `5%` 的空闲样本，工作时平均利用率也只有 `31.8%`。同期最大显存占用仅
`5.85 GiB / 24 GiB`。因此除了增加任务覆盖时间，还需要确认 TreeHeap 训练是否能通过
增大 micro-batch 提高单位墙钟时间吞吐。

本 canary 只优化计算引擎，不评价 D12 算法收益，也不改变正在运行的 D12-R1 合同。

## 2. 严格合同

依次测试 `batch=16/32/64`。每档均从相同 source checkpoint 和 D10-R1 warm start
重新初始化，使用相同数据、seed、固定草图、学习率、深度轮换和 `predictive_weight=0.53`。
每档的 `probe-only` 与 `predictive-gradient` 两臂各运行 300 steps。

前 50 steps 视为启动和缓存预热区间。主吞吐为 step 50 到 step 300 的：

```text
target_tokens_per_second = delta(processed_target_tokens) / delta(train_elapsed_seconds)
```

独立的 `nvidia-smi` 采样器每两秒记录利用率、显存、功耗、温度与功率上限。

## 3. 安全边界

候选 batch 必须满足：

1. 两臂完成 300 steps，梯度与最终指标有限；
2. 功率上限始终不超过 270.5W；
3. 峰值显存不超过 20 GiB，为 24 GiB 卡保留至少约 4 GiB 余量；
4. 峰值温度不超过 80°C；
5. checkpoint/reload 合同通过。

某个较大 batch OOM 时保留失败记录，不解除功率或频率限制，也不继续扩大 batch。

## 4. 选择规则

在通过安全边界的候选中，选择两臂合计 `target_tokens_per_second` 最高者，作为后续 D12
同类训练的默认 micro-batch。若大 batch 没有提高吞吐，则保留较小档。吞吐 canary 的
短程 NLL、BLEU、MSE 和重复率只用于确认数值未损坏，不用于判断模型优劣。

该结果只适用于当前 TreeHeap-63M/D12 计算图；模型宽度、序列长度或结构变化后需要重新
校准。
