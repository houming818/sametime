# TreeHeap E02：106M 批量与输入预取吞吐实验

日期：2026-09-07

Claim：`S3-EPOCH-REPEAT-THROUGHPUT-E02`

状态：预注册；CPU smoke 通过后排在 E01 配对汇总之后执行。

## 1. 问题

E01 固定使用 `batch=64`，优先保护 63M/106M 配对合同。该选择并不证明它充分利用了
RTX 3090。E02 不继续第三遍语料，也不改变任何正式 checkpoint，只用同一个不可变 D11
106M checkpoint 测量更大的 batch 与有界输入预取能否提高有效 token/s。

## 2. 固定合同

四个独立 case 均从 D11 `treeheap-106m` 的
`step=25,000/cursor=400,488` checkpoint 重新加载，并使用相同 source、warm start、
语料、seed、方向函数、深度轮换和学习率：

| case | batch | 后台预取队列 |
|---|---:|---:|
| `b64-raw` | 64 | 0 |
| `b64-prefetch2` | 64 | 2 batches |
| `b96-prefetch2` | 96 | 2 batches |
| `b128-prefetch2` | 128 | 2 batches |

每个 case 严格运行 200 optimizer steps。预取线程只提前完成文件读取、解析、方向选择和
SentencePiece 编码；batch 顺序与内容不变，GPU 计算和优化器语义不变。所有产出的模型更新
均为临时吞吐探针，完成 reload 校验后删除大 checkpoint，不进入后续训练。

## 3. 测量与选择

从去掉首个 trace 区间后的稳定窗口计算：

- target tokens/s 与 examples/s；
- 相对 `b64-raw` 的吞吐倍数；
- 峰值显存、功率和温度；
- loss/gradient 有限性与保存后 reload NLL 差。

仅在所有安全门通过且峰值显存不超过 `23,000 MiB` 的 case 中，选择稳定 token/s 最高者。
该选择只决定未来实验的默认吞吐配置，不构成模型质量 Claim。

## 4. 否证与安全门

- 任一 OOM、CUDA allocator/assert、NVIDIA Xid、NaN/Inf：停止并保留失败 case；
- 功率限制高于 `270.5W`：不得启动该 case；
- checkpoint、optimizer、哈希或 reload 失败：整项实验不授权配置变更；
- 预取导致顺序、数量或异常传播改变：CPU smoke 失败，不进入 GPU 队列；
- 无论吞吐结果如何，不自动启动 150M 或第三遍语料。

E02 排在 E01 的任务 364 之后，避免与正式配对训练争用 GPU，也避免中途修改 E01 的
`batch=64` 合同。
