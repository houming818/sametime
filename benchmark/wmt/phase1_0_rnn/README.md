# Phase 1.0: Vanilla RNN Seq2Seq

> 路径: `benchmark/wmt/phase1_0_rnn/`

## 核心思想
使用最基本的 RNN (tanh) 实现 Encoder-Decoder。
理解 RNN 的记忆原理：每个时间步的隐状态 h_t 由上一步 h_{t-1} 和当前输入 x_t 共同决定。

## RNN 公式
- h_t = tanh(W_hh · h_{t-1} + W_ih · x_t + b)

## 关键设计
- **Encoder**: 2 层双向 RNN，hidden_size=256
- **Decoder**: 2 层单向 RNN，hidden_size=256
- **训练**: teacher forcing + gradient clipping

## 致命弱点
长句中 tanh(W) 连乘 → 梯度指数衰减 → 梯度消失。
这就是 Phase 1.1 引入 LSTM (门控 + 细胞状态) 的直接原因。

## 张量形状
| Tensor | Shape | 说明 |
|--------|-------|------|
| `src`  | (B, S) | 源 token ids |
| `enc_out` | (B, S, H*2) | BiRNN 各时刻输出 |
| `hidden` | (N, B, H) | Encoder 最后 hidden (sum 双向) |
| `tgt_in` | (B, T) | teacher forcing 输入 |
| `logits` | (B, T, V) | 每步预测 |

## 如何运行
```bash
make wmt_phase1_0
```

## 预期结果
- BLEU < LSTM (梯度消失更严重)
- 训练 loss 下降比 LSTM 慢
- 学习价值 > 数值表现

## 对照分析实验

### 对照分析实验 start

**实验目的**：控制 hidden_size 变量，观察 RNN Seq2Seq 的容量-质量关系。

**固定变量**：

| 变量 | 值 |
|---|---|
| `epochs` | 5 |
| `lr` | 1e-3 |
| `embed` | = hidden |
| `layers` | 2 |
| `dropout` | 0.3 |
| `batch_size` | 64 |
| `seed` | 42 |
| `optimizer` | Adam |
| `dataset` | IWSLT14 de-en |
| `device` | RTX 3090 (24GB) |

**自变量**：`hidden ∈ {16, 32, 64, 128, 256, 512, 1024}`

**实验脚本**：`./exp_hidden_scale.sh`（运行在 io.grepcode.cn Docker）

---

#### 已完成组 (2026-04-30)

| hidden | params | epoch-0 loss | best BLEU | epoch-0 BLEU | epoch-4 BLEU | 耗时 |
|---|---|---|---|---|---|---|
| 16 | 3,063,052 | 7.038 | 2.13 (ep4) | 0.00 | 2.13 | 285s |
| 32 | 6,076,620 | 6.631 | 2.94 (ep4) | 1.05 | 2.94 | 290s |
| 64 | 12,125,260 | 6.240 | 2.97 (ep0) | 2.97 | 2.55 | 307s |
| 128 | 24,308,556 | 5.967 | 2.71 (ep0) | 2.71 | 2.17 | 379s |
| 256 | 49,019,212 | 5.736 | 2.32 (ep2) | 2.02 | 2.24 | 528s |
| 512 | 99,816,780 | 5.691 | **3.02 (ep2)** | 1.84 | 2.48 | 895s |
| 1024 | 206,916,940 | 5.981 | 2.42 (ep1) | 1.57 | 1.60 | 1741s |

**原始 jsonl 数据**：

| 文件 | 行数 | 说明 |
|---|---|---|
| `results/metrics_h16.jsonl` | 5 | H=16, 5 epochs |
| `results/metrics_h32.jsonl` | 5 | H=32, 5 epochs |
| `results/metrics_h64.jsonl` | 5 | H=64, 5 epochs |
| `results/metrics_h128.jsonl` | 5 | H=128, 5 epochs |
| `results/metrics_h256.jsonl` | 5 | H=256, 5 epochs |
| `results/metrics_h512.jsonl` | 5 | H=512, 5 epochs |
| `results/metrics_h1024.jsonl` | 5 | H=1024, 5 epochs |

存储于 io: `/data/homecicd/sametime/results/`

#### 初步结论

1. **BLEU 天花板 ≈ 3.0**：所有配置的最高 BLEU 不超过 3.02（H=512, epoch 2）。无 Attention 的 Seq2Seq 在 IWSLT14 de-en 上存在硬上限。
2. **隐藏层扩大有最佳点**：BLEU 随 hidden 增大先升后降——H=512 (100M) 达峰 3.02，H=1024 (207M) 反而退到 2.42。**过参数化导致过拟合**。
3. **小模型收敛慢但持续进步**：H=16/32 的 BLEU 随 epoch 单调上升，epoch-4 才达到 peak。大模型 (H≥128) 在 epoch 0-2 即达峰后衰退。
4. **Loss 持续下降，BLEU 不升反降**：所有配置 loss 均递减，但多个 (H=128/512/1024) 的 BLEU 在中期 epoch 后下降。这是**过拟合**的典型信号——模型在 teacher forcing 上愈学愈好，但泛化到 Greedy Decode 时反而退化。
5. **参数量与时间基本线性**：H=16 (3M) 285s → H=1024 (207M) 1741s，每 10M 参数约 80 秒。

#### 预测 3 验证：epoch 深度 (2026-05-01)

H=512 ep20 和 H=1024 ep20 对照：

| 实验 | best BLEU | peak epoch | epoch 19 BLEU | 耗时 |
|---|---|---|---|---|
| H=512 ep20 | 3.02 | 2 | 1.35 | 56 min |
| H=1024 ep20 | 2.42 | 1 | 1.94 | 135 min |

**结论**：预测 3 成立。epoch 不改变 K_lang——更深 epoch 导致更严重过拟合。H=1024 在 epoch 6 后 loss 拐头上升。350W 功耗限制（前主人刷 390W BIOS 导致 Xid 79 掉卡）验证有效。

### 对照分析实验 end
