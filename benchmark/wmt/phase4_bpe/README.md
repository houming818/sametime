# Phase 4: BPE / SentencePiece（2016）

> 路径: `benchmark/wmt/phase4_bpe/`

## 核心思想
从 word-level 替换为 **subword-level**（Byte-Pair Encoding），
解决 OOV（out-of-vocabulary）问题，词表从~30k 降到~8k 且不再有 `<unk>`。

## 关键改动
1. 训练 SentencePiece 模型（联合分词：源/目标共享词表或各自训练）
2. 替换 `base/dataset.py` 中的 `Vocab` 为 SentencePieceProcessor
3. 模型结构完全不变——只换 tokenizer 不换架构

## BLEU 提升来源
| 原因 | 说明 |
|------|------|
| 消除 `<unk>` | 子词总能覆盖未见过的词 |
| 更稳定的 embedding | 高频子词训练更充分 |
| 更好的泛化 | 形态学相似的词共享子词 |

## 运行

```bash
# 1. 训练 SentencePiece 模型
make wmt_phase4_bpe
```
