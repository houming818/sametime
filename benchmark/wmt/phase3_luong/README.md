# Phase 3: Luong Attention + Beam Search（2015）

> 路径: `benchmark/wmt/phase3_luong/`

## 核心思想
1. **Luong Attention**: 相比 Bahdanau（additive），Luong 提出更简单的 dot/general score，且 attention 在 decoder output 上计算而非 hidden state（"global attention"）
2. **Beam Search**: 解码时保留 top-k 个候选路径，而不是只取最优 token，显著提升 BLEU

## Attention 变体（Luong 2015）
| 类型 | 公式 | 说明 |
|------|------|------|
| dot | $score = h_t^\top \bar{h}_s$ | 最简单 |
| general | $score = h_t^\top W \bar{h}_s$ | 可学习的线性变换 |
| concat | $score = v^\top \tanh(W [h_t; \bar{h}_s])$ | 类似 Bahdanau |

## Beam Search
- beam_size=1 → greedy
- beam_size=3~5 → 常见配置
- length penalty 避免偏好短句

## 与 Phase 2 对比
| 指标 | Phase 2 | Phase 3 |
|------|---------|---------|
| BLEU | ~10-15 | ~15-22 |
| 解码速度 | 快 | 慢（beam_size 倍） |
| 长句 | 中等 | 好（beam search 更稳定） |

## 运行

```bash
make wmt_phase3

# 指定 beam_size
make wmt_phase3 ARGS="--beam 5"
```
