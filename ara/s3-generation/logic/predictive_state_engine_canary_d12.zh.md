# D12-E1：消费级 GPU 吞吐引擎 Canary

日期：2026-09-04

Claim：`S3-PREDICTIVE-STATE-ENGINE-CANARY-D12-E1`

状态：正式 canary 完成；当前计算图推荐 `batch=64`。

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

## 5. 正式结果

`io.grepcode.cn` 上 taskd 任务 `360` 完成全部三档，每档两臂各 300 steps。所有有限值、
冻结 source、模型更新、预测头更新和重载检查均通过；功率上限始终为 270W，没有 OOM
或 CUDA/NVIDIA 异常。总墙钟时间为 1,079.2 秒。

| batch | 合计 target token/s | 相对 batch 16 | 峰值显存 | 峰值温度 | 峰值功耗 |
|---:|---:|---:|---:|---:|---:|
| 16 | 814.03 | 1.00x | 2.65 GiB | 73°C | 232.38W |
| 32 | 1,342.57 | 1.65x | 3.23 GiB | 74°C | 236.24W |
| 64 | 2,189.63 | 2.69x | 5.08 GiB | 74°C | 240.49W |

`batch=64` 相对 `batch=32` 又提高约 `63.1%` 吞吐，相对 `batch=16` 提高约 `169%`，且
只使用约五分之一显存。依预注册规则，当前 D12/TreeHeap-63M 计算图的默认 micro-batch
更新为 `64`。

GPU 利用率采样均值没有随 batch 同步升高，三档约为 `29-31%`，但 token/s 明显增加。
这说明当前 `nvidia-smi utilization` 更适合判断空闲区间，不能单独代表本计算图的有效
吞吐；训练调度应优先用 token/s，并把利用率、显存、功耗和温度作为并列约束。

本档仍未找到硬件饱和点。`batch=64` 峰值显存只有 5.08 GiB，后续可预注册相邻的
`96/128` 阶梯，但不能把本结果外推成更大 batch 必然继续提速。

正式 evidence：`../evidence/s3_predictive_state_sketch_d12/engine_canary_20260904/`。
