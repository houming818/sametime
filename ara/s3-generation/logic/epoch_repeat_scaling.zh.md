# TreeHeap Epoch Repeat Scaling：完整重复语料尺度曲线

日期：2026-09-04

Claim：`S3-EPOCH-REPEAT-SCALING-E01`

状态：正式配对续训完成；NLL 容量分离得到支持，生成侧为混合结果。

## 1. 问题

D10 的 63M TreeHeap 已经把 `NioClean-ZHEN-S098-7M-v2` 的 `7,304,358`
行完整读取一遍。D11 随后从同一个 D10 checkpoint 建立两条续训臂：原 63M 系统与
增加零门控 384 维残差 TreeHeap 通道的 106M 系统。两臂都训练了 25,000 steps，游标
均为 `400,488`，即只完成第二遍语料的约 `5.48%`。

D11-R1 在这个固定短预算下得到的 106M 相对 NLL 收益只有 `0.004722`。该结果有效地
否定了“新增容量会在 25K steps 内产生明显收益”，但不能回答新增 43M 参数在完成一遍
重复训练后是否仍然无效。

本实验不增加到 150M，也不引入 D12 预测草图。唯一目标是把 D11 的两条学习曲线沿同一
数据游标继续到第二遍语料结束。

## 2. 配对合同

两臂分别从以下 D11 `checkpoint_latest.pt` 恢复：

| arm | 参数量 | 起始 step | 起始 cursor |
|---|---:|---:|---:|
| `treeheap-63m` | 62,748,179 | 25,000 | 400,488 |
| `treeheap-106m` | 105,965,594 | 25,000 | 400,488 |

共同保持：

- 同一 source checkpoint、D10 warm start、SentencePiece 与平行语料哈希；
- 同一 seed、ownership seed、深度轮换、损失函数和学习率 `1e-4`；
- 同一数据顺序、翻译方向函数、起止游标和 batch；
- 继承各自 AdamW optimizer state，不重新初始化动量；
- source encoder 继续冻结；
- target 不进入 source TreeHeap。

本轮是已经开始的第二遍语料，因此继续沿用 D11 的原始顺序和方向函数，不能在中途改变
排列。下一完整遍才允许使用另行预注册的确定性分块重排，并翻转每行的双语方向。

## 3. 算力合同

正式训练使用 `io.grepcode.cn` 的 RTX 3090，保留 `270W` 功率限制。D12-E1 已证明同族
模型在 `batch=64` 时吞吐约为 batch16 的 `2.69x`，但正式启动前仍须用真实 106M D11
checkpoint 做恢复冒烟。

```text
batch: 64
target cursor: 7,304,358
segment: at most 10,000 optimizer steps per Python/CUDA process
evaluation/checkpoint receipt: every segment
formal arms: 63M then 106M, serialized by nio-taskd
```

切成 segment 只为释放 CUDA 进程状态和提供可恢复 checkpoint，不重置模型、optimizer、
全局 step、数据 cursor 或学习率。

## 4. 预测与判读

### P0：恢复合同

真实 checkpoint 加载后，模型参数 SHA-256 必须与 checkpoint 一致；optimizer state 必须
成功加载；第一次更新前的 NLL 必须有限。

### P1：完整暴露

两臂都必须到达 cursor `7,304,358`，并具有连续的 segment receipts。每条曲线至少覆盖
全语料的 25%、50%、75% 和 100% 位置。

### P2：容量动力学

若 106M 的短预算劣势主要来自新增通道训练不足，则随着暴露增加，
`NLL(63M)-NLL(106M)` 应总体向正方向分离，或者生成/结构指标出现稳定互补收益。
若差值持续接近零、反向扩大，或新增通道增益保持无效，则当前容量扩展配方不受支持。

P2 是曲线解释，不是运行中止门。NLL、BLEU、重复率和某个 wake 的非单调变化全部保留，
不得据此提前终止。

### P3：结构与产品观察

在最终 cursor 记录：

- Test NLL/PPL；
- 固定生成样本、token BLEU-4、非空率、重复率；
- native/shuffle/zero 输入因果；
- route coverage、overlap、slot variance；
- 106M 的 residual logit gain；
- checkpoint reload 一致性。

这些指标用于区分“普通语言骨架继续学习”和“新增 TreeHeap 容量产生结构贡献”。本实验
不声称形成逻辑、世界模型或产品级翻译。

## 5. 停机规则

训练必须跑到固定语料边界。只允许在以下情况下停止：

- OOM、CUDA allocator/assert、NVIDIA Xid；
- loss/gradient 出现 NaN 或 Inf；
- checkpoint、optimizer、哈希或 reload 损坏；
- 数据游标不前进、越界或两臂数据合同不一致；
- 3090 功率限制超过 `270.5W`。

完成本轮只产生一条完整 repeat scaling curve，不自动启动 150M，也不自动开始第三遍语料。

## 6. 真实 checkpoint 冒烟

`io` taskd 任务 `361` 已完成。63M 与 106M 均从 D11 的
`step=25,000/cursor=400,488` 恢复，使用 `batch=64` 各训练 20 steps，最终同步到
`step=25,020/cursor=401,768`。

两臂的 optimizer state 均成功加载；loss 与梯度有限；source encoder 哈希不变；模型哈希
发生变化；route 的 owner/argmax coverage 均为 `1.0`；保存后 reload NLL 差为 `0`。
106M 没有出现 OOM 或 CUDA 故障。因此冒烟授权按本文件固定合同进入完整配对续训。

冒烟 evidence：`../evidence/s3_epoch_repeat_scaling_e01/smoke_seed11301/`。

## 7. 正式启航

`2026-09-04` 已在 `io` 依次提交：

```text
taskd 362: treeheap-63m pass-2 completion
taskd 363: treeheap-106m pass-2 completion (depends on 362)
taskd 364: paired curve comparison
```

63M 首批状态为有限值，3090 功率限制 `270W`，训练中观测约 `210W / 67C / 7.2GiB`。
预计两臂和最终汇总共需约 `2.5--3.5` 天；实际时间以 segment 吞吐为准。

## 8. 正式结果

两臂均从 `step=25,000/cursor=400,488` 连续训练到
`step=132,873/cursor=7,304,358`。每臂具有 11 个连续 segment receipts，source、
warm start、语料、batch、方向与起止游标合同全部匹配；有限值、source frozen、optimizer
恢复、模型更新、游标边界和 reload 门全部通过。

最终配对指标为：

| 指标 | 63M | 106M | 106M 相对变化 |
|---|---:|---:|---:|
| Test NLL | 3.828686 | 3.775919 | 改善 0.052767 |
| token BLEU-4 | 6.538087 | 5.642579 | 降低 0.895508 |
| 重复率 | 0.015677 | 0.048399 | 增加 0.032722 |
| 非空率 | 0.968750 | 0.953125 | 降低 0.015625 |

D11-R1 的 25K 短预算中，106M 的 NLL 收益仅为 `0.004722`；完成第二遍语料后收益扩大到
`0.052767`。因此，当前证据支持“新增容量的 NLL 收益依赖足够训练暴露”，并否定了仅凭
25K 短预算判断该容量永远无效的做法。

这不是完整产品收益。106M 的固定样本 BLEU 更低、重复率更高，新增 residual logit gain
最终为 `-0.999959`。因此本实验只支持容量与训练预算之间的 NLL 动力学，不授权 150M，
也不授权第三遍语料。生成侧差异需要更大的固定生成集或独立种子确认，不能由 64 个样本
直接推断为稳定退化。

正式 evidence：`../evidence/s3_epoch_repeat_scaling_e01/formal_seed11301/`。
