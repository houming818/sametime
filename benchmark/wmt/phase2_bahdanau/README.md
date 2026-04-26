# Phase 2: Bahdanau Additive Attention（2015）

> 路径: `benchmark/wmt/phase2_bahdanau/`

## 核心思想
在 Phase 1 基础上，Decoder 每步"看"一遍整个源句子，而不是只依赖最后 hidden state。

## 公式（Bahdanau / Additive Attention）
1. 对齐分数: $e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)$
2. 注意力权重: $\alpha_{ij} = \text{softmax}(e_{ij})$
3. 上下文向量: $c_i = \sum_j \alpha_{ij} h_j$
4. Decoder 输入: $[y_{i-1}; c_i]$（拼接）

## 与 Phase 1 的对比
| 指标 | Phase 1 (no attn) | Phase 2 (Bahdanau) |
|------|-------------------|---------------------|
| BLEU | < 5 | ~10-15 |
| 长句翻译 | 差 | 好（attention 可聚焦不同位置） |
| 训练速度 | 快 | 稍慢（多出 attention 计算） |

## 运行

```bash
make wmt_phase2
```
