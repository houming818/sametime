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
