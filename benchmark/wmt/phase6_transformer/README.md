# Phase 6: Transformer（Vaswani 2017）

> 路径: `benchmark/wmt/phase6_transformer/`

## 核心思想
**"Attention Is All You Need"** — 完全舍弃 RNN，只用自注意力 + FFN。

## 关键组件
| 模块 | 说明 |
|------|------|
| Multi-Head Self-Attention | 每个位置关注所有位置，多头并行 |
| Positional Encoding | 正弦波编码，给序列注入位置信息 |
| Feed-Forward Network | 两层线性 + ReLU，每个位置独立 |
| Residual + LayerNorm | 每个子层后 Add & Norm |
| Label Smoothing | 防止过拟合，提升泛化 |
| Warmup LR | 学习率先升后降，稳定训练 |

## 与 RNN 的关键对比
| 指标 | RNN (Phase 3) | Transformer (Phase 6) |
|------|---------------|-----------------------|
| 并行度 | 串行（时间步依赖） | 完全并行 |
| 训练速度 | 慢 | 快（GPU 利用率高） |
| 长程依赖 | 差（梯度消失） | 好（直接 attention） |
| BLEU (IWSLT14) | ~15-22 | ~25-35 |
| 参数量 | 小 | 大（但效率更高） |

## 运行
```bash
make wmt_phase6
```
