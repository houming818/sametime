# C12-R1：严格多层 READ 的多 seed 复制

日期：2026-08-24

Claim：`S3-MULTILEVEL-READ-ABLATION-C12-R1`

状态：正式队列运行中，io taskd `315`。

## 1. 背景

C12 seed `10101` 的 25,000-step 正式实验满足同初始化、同数据流和有限数值合同，
但机制门 P1 失败：leaf-only 有明显损失，而“至少两个 non-leaf depth 各自不可
替代”没有成立。单 seed 不足以判断这是稳定规律还是初始化偶然。

C12-R1 不改模型、数据、loss、优化器或干预定义，只增加 seed `10102/10103`。
它与 `IDEA-001` 隔离，不使用分层目标。

## 2. 实验矩阵

每个 seed 串行训练三个臂：

| arm | READ | 额外 bottom-up pass |
|---|---|---|
| `c10` | learned STOP probability container | 无 |
| `read` | 强制多层 residual READ | 无 |
| `read_up` | 强制多层 residual READ | shared `K_up` |

固定参数：200,000 train rows、1,000 valid、1,000 test、25,000 updates、batch 16、
AdamW、LR `0.002`、gradient clipping `1.0`。父 checkpoint、tokenizer、数据和代码
均沿用 C12。

## 3. 预注册门

每个 seed 首先满足 P0：

- 三臂父状态 hash、row hash 与训练流 hash 一致；
- `read/read_up` 初始 READ 参数 hash 一致；
- 梯度、loss 和 checkpoint 均有限、可重载。

机制复制使用原 C12 门：

- `read` 的 leaf-only NLL 至少比 native 高 `0.05`；
- `read` 至少两个 non-leaf depth 消融各使 NLL 增加 `0.01`；
- `read_up` 的 bypass-`K_up` delta 独立报告。

只有三个 seed 中至少两个完整通过上述 READ 机制门，才把 C12 的单 seed 失败降级
为不稳定结果。若新增两个 seed 仍都失败，则把“逐深度不可替代”记录为稳定负
结果；这不否定组级分布式贡献。

## 4. 停止条件

OOM、CUDA/GPU 故障、NaN/Inf、hash 不一致、恢复合同不一致或 checkpoint 损坏
立即停止。该复制不授权 100M Pretrain 或产品发布。
