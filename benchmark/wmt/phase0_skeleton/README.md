# Phase 0: 实验底座骨架

> 路径: `benchmark/wmt/phase0_skeleton/`

## 目标
固定数据、评估、训练循环，先用一个极简模型跑通全流程。

## 关键设计
- **数据**: IWSLT14 De-En，word-level tokenizer，`Vocab` 统一管理
- **模型**: 一个最简单的"复制"模型（输入什么输出什么），仅验证管线
- **评估**: sacreBLEU
- **训练**: teacher forcing

## 如何运行

```bash
# 在 ni 上开发，在 io 上通过 DOCKER_HOST 运行
make wmt_phase0
```

## 目录结构
```
phase0_skeleton/
├── README.md         ← 本文档
└── train.py          ← 完整训练+评估脚本
```
## 学习要点
- IWSLT14 的数据格式（translation 字段）
- word-level vocab 的构建与 OOV 处理
- collate_fn 如何对齐 batch 长度
- BLEU 的数值范围与意义（很低也正常，因为 baseline 是复制）
