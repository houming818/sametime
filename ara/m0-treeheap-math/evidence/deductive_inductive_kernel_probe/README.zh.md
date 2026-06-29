# README（中文读者版）

对应英文文件：`ara/m0-treeheap-math/evidence/deductive_inductive_kernel_probe/README.md`  
所属主题：M0 TreeHeap Math（数学地基）

> 说明：这是递归生成并人工组织过的中文读者版。它保留原文件的 ARA 用途、claim/evidence 边界和 reviewer 读法；命令、ID、路径、指标名通常保留英文，方便和原文及实验输出对照。

## 文件用途

本文件说明当前目录如何组织、如何从 claim 找到 evidence，以及 reviewer 应该如何进入这层 ARA。

## 阅读要点

- M0 的目标不是直接做语言任务，而是先证明 TreeHeap 的数学操作是否闭合、可读、可训练。
- 关键问题包括 plus、kernel convolution、diff/distance、soft lifting、algebraic decoder。
- 判断时要区分演绎证明和归纳学习：代数恒等式可以精确验证，概率 kernel 需要实验和 KL/accuracy 指标。
- 读取 evidence 时优先看 summary.json / trace.jsonl / README，再回到 logic/claims.md 更新 claim 状态。

## Reviewer 检查方式

- 证据目录里的数字是 claim 状态变化的依据。
- 如果 README 和 summary.json 冲突，以可复现 summary/trace 为准，并修正文档。

## 与英文原文的关系

- 英文原文保留为 canonical source，中文版本用于中文读者快速理解和审阅。
- 涉及实验数字、命令、文件路径、claim ID 时，应以英文原文和 evidence 目录中的机器输出为最终校验来源。
- 如果后续英文原文有 claim 状态更新，本文件也应同步更新。
