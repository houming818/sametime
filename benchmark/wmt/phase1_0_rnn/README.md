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

| hidden | params | epoch-0 loss | epoch-0 BLEU | epoch-4 loss | epoch-4 BLEU | best BLEU | 时间 |
|---|---|---|---|---|---|---|---|
| 128 | 24,308,556 | 5.967 | 2.71 | 4.841 | 2.17 | 2.71 (epoch 0) | ~2min |
| 256 | 49,019,212 | 5.736 | 2.02 | 4.561 | 2.24 | 2.32 (epoch 2) | ~5min |

**原始数据**：

```
# H128
{"epoch":0,"loss":5.967,"bleu":2.71}
{"epoch":1,"loss":5.287,"bleu":2.44}
{"epoch":2,"loss":5.070,"bleu":2.26}
{"epoch":3,"loss":4.936,"bleu":2.52}
{"epoch":4,"loss":4.841,"bleu":2.17}

# H256
{"epoch":0,"loss":5.736,"bleu":2.02}
{"epoch":1,"loss":5.050,"bleu":2.12}
{"epoch":2,"loss":4.817,"bleu":2.32}
{"epoch":3,"loss":4.668,"bleu":2.17}
{"epoch":4,"loss":4.561,"bleu":2.24}
```

#### 初步结论

1. **容量翻倍，BLEU 不变**：H128→H256 参数量翻倍（24M→49M），但 BLEU 在 2.0~2.7 之间没有显著提升。这符合无 Attention 的 Seq2Seq 信息瓶颈理论——hidden 再大也无法绕过"上下文向量瓶颈"。
2. **最佳 BLEU 出现在早期 epoch**：H128 的 best BLEU 在 epoch 0（2.71），之后不升反降。可能在小容量模型中发生了过拟合。
3. **H256 loss 收敛更快**：final loss 4.56 vs H128 的 4.84，但 BLEU 在 epoch 2 达到峰值后就停滞了。

**待补数据**：H=16/32/64/512/1024 (io GPU 离线，需重启后执行 `./exp_hidden_scale.sh`)

### 对照分析实验 end
