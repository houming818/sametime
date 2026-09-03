# D11-R1：固定预算容量阶梯恢复

日期：2026-09-03

Claim：`S3-STRUCTURAL-PROTOCOL-CAPACITY-LADDER-D11-R1`

状态：正式实验完成；实验有效，第一档容量扩容 Claim 不受支持。

## 1. 恢复原因

D11 的 `TreeHeap-63M` 正式臂已经完整训练 25,000 steps，但 depth 7 的生成非空率为
`63/64 = 0.984375`。原 runner 把非空率严格等于 `1.0` 设为 arm 退出门，随后 shell 的
`set -e` 阻止了 `TreeHeap-106M` 正式臂启动。

这保留为 D11 的真实失败记录。它说明原训练编排不适合回答容量问题，不说明 106M
容量无效，因为 106M 正式训练并未发生。

## 2. 原则修正

训练曲线允许非单调。NLL、BLEU、重复率和非空率只在固定 wake 记录，不用于中途停止。
每个 arm 必须跑满预注册预算，之后才进行比较。

训练期间只有以下硬故障允许停止：

1. OOM、CUDA allocator/assert、NVIDIA Xid 或 GPU 不可用；
2. loss、梯度或模型状态出现 NaN/Inf；
3. checkpoint 无法重载、哈希损坏或 evidence 不完整；
4. step/cursor 长时间不前进；
5. RTX 3090 的 270W 功率限制失效；
6. 输入数据、seed、warm start 或模型合同与预注册不一致。

任何阶段性质量波动都不属于硬停止条件。

## 3. 固定实验

D11-R1 不重跑已经完成的 63M 臂，只补齐缺失的 106M 臂：

```text
baseline: D11 TreeHeap-63M formal, 25,000 steps, cursor 400,488
scale: TreeHeap-106M, zero-gated 384-dimensional residual TreeHeap channel
warm start: 同一 D10-R1 best checkpoint
source: 同一冻结 C12 READ checkpoint
data: NioClean-ZHEN-S098-7M-v2
seed: 11101
ownership seed: 11102
batch size: 16
training steps: 25,000
wake interval: 5,000
device: io RTX 3090 24GB, power limit <= 270.5W
```

启动前必须重新确认 D11 smoke 的 step-0 NLL 差为 `0`，两臂参数量分别约为
`62.75M/105.97M`，并通过 CPU 递归通道合同。

## 4. 完成后评价

先判断实验是否有效：同 source、warm start、data、seed、steps、cursor，step-0 NLL 差
不超过 `1e-8`；两臂均完成数值有限、输入因果、结构覆盖、冻结 source、参数更新和
checkpoint reload。

只有固定预算全部完成后，才评价扩容收益：

```text
NLL(63M) - NLL(106M) >= 0.03
BLEU4(106M) - BLEU4(63M) >= -0.50
repetition(106M) - repetition(63M) <= 0.02
nonempty(106M) >= nonempty(63M)
```

不满足质量门只表示第一档容量 Claim 不受支持，不把已完成任务标记成运行失败。无论结果
如何，D11-R1 都不自动启动 150M；下一档必须另行预注册，并继续满足单张 24GB 消费级
GPU 的产品方向。

## 5. 正式结果

`io.grepcode.cn` 上的恢复任务 `354` 已跑满 25,000 steps。两臂使用相同 source、
warm start、数据、seed、cursor 和训练步数，step-0 NLL 差为 `0`。两臂的数值有限性、
输入因果、TreeHeap 结构覆盖、冻结 source、可训练参数更新和 checkpoint reload 均通过。

| 指标 | TreeHeap-63M | TreeHeap-106M | 106M 相对变化 |
|---|---:|---:|---:|
| Test NLL | 4.009855 | 4.005133 | 改善 0.004722 |
| token BLEU-4 | 8.450974 | 8.436652 | -0.014323 |
| adjacent repetition | 0.048083 | 0.048687 | +0.000604 |
| nonempty rate | 0.984375 | 0.984375 | 0 |
| 总参数量 | 62,748,179 | 105,965,594 | +43,217,415 |
| 可训练参数量 | 27,381,254 | 70,598,669 | +43,217,415 |

106M 满足 BLEU、重复率和非空率的非劣门，但 NLL 收益 `0.004722` 远低于预注册的
`0.03`。因此结论是 `first_scale_rung_not_supported`：增加 43.2M 可训练参数没有破坏
系统，也形成了可训练、可重载的额外 TreeHeap 通道，但在相同 25K 数据/步数预算下
几乎没有转化为任务质量。

这不是“更多参数永远无效”的证明，也不能排除更长训练后曲线再次分离；它只否定
“在当前 warm start、数据流和固定 25K 预算下，106M 会明显优于 63M”这一具体 Claim。
根据预注册约束，不自动启动 150M。

正式证据：`../evidence/s3_structural_protocol_capacity_ladder_d11/formal_seed11101/`，
核心比较文件为 `comparison_r1.json`。
