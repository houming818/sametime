# Phase 1.1: LSTM Seq2Seq

> 路径: `benchmark/wmt/phase1_1_lstm/`

## 核心思想
在 Phase 1.0 vanilla RNN 基础上，用 LSTM 的门控 + 细胞状态机制解决梯度消失。
Encoder 用 BiLSTM 编码源句，Decoder 用 LSTM 逐词生成翻译。

## LSTM vs RNN

| | Vanilla RNN (Phase 1.0) | LSTM (Phase 1.1) |
|---|---|---|
| 状态 | h_t = tanh(W·[h_{t-1},x_t]) | c_t = f_t⊙c_{t-1}+i_t⊙c̃_t, h_t = o_t⊙tanh(c_t) |
| 梯度路径 | 连乘 tanh(W) → 消失 | 加性直通 → 保持 |
| 细胞状态 | 无 | c_t: 梯度高速通道 |
| 参数 | H×H (单权重矩阵) | H×H×4 (三扇门+候选) |
| 长句性能 | 差 (信息瓶颈) | 优于 RNN，仍逊于 Attention |

## 关键公式
- f_t = σ(W_f·[h_{t-1}, x_t])
- i_t = σ(W_i·[h_{t-1}, x_t])
- o_t = σ(W_o·[h_{t-1}, x_t])
- c̃_t = tanh(W_c·[h_{t-1}, x_t])
- c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
- h_t = o_t ⊙ tanh(c_t)

## 如何运行
```bash
make wmt_phase1_1
```

## 预期结果
- BLEU 显著高于 Phase 1.0 vanilla RNN
- 训练更稳定 (gradient clip 辅助)
- 仍受"上下文向量瓶颈"限制 → Phase 2 引入 Attention
