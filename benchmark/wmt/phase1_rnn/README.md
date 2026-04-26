# Phase 1: RNN Seq2Seq（2014 Sutskever）

> 路径: `benchmark/wmt/phase1_rnn/`

## 核心思想
Encoder 将源句子编码为固定维度的"上下文向量"（最后 hidden state），
Decoder 从这个向量开始逐词生成。

## 关键公式
- Encoder: $h_t = \text{BiLSTM}(x_t, h_{t-1})$
- 上下文向量: $c = [\overrightarrow{h}_T; \overleftarrow{h}_T]$
- Decoder: $s_t = \text{LSTM}(y_{t-1}, s_{t-1}, c)$
- 输出: $p(y_t | y_{<t}, c) = \text{softmax}(W s_t)$

**痛点**: 长句性能差——最后 hidden state 是信息瓶颈。

## 张量形状
| Tensor | Shape | 说明 |
|--------|-------|------|
| `src`  | (B, S) | 源 token ids |
| `enc_out` | (B, S, H*2) | BiLSTM 各时刻输出 |
| `hidden` | (2*2, B, H) | Encoder 最后 hidden/cell (=num_layers*2, B, H) |
| `tgt_in` | (B, T) | 目标 token ids (teacher forcing) |
| `logits` | (B, T, V) | 每步预测 logits |

H = hidden_size, S = src_len, T = tgt_len, V = vocab_size

## 如何运行

```bash
# train
make wmt_phase1

# eval (after training, resume from checkpoint)
make wmt_phase1 ARGS="--checkpoint checkpoints/phase1.pt"
```

## 预期结果
- BLEU 很低（< 5），因为无 attention + word-level
- 短句翻译质量略好于长句
- 训练速度较快（纯 RNN，无额外计算）
