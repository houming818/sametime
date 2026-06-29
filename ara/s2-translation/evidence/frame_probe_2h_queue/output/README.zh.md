# README（中文读者版）

对应英文文件：`ara/s2-translation/evidence/frame_probe_2h_queue/output/README.md`  
所属主题：S2 Translation（折叠、图构建与翻译前结构）

> 说明：这是递归生成并人工组织过的中文读者版。它保留原文件的 ARA 用途、claim/evidence 边界和 reviewer 读法；命令、ID、路径、指标名通常保留英文，方便和原文及实验输出对照。

## 文件用途

本文件说明当前目录如何组织、如何从 claim 找到 evidence，以及 reviewer 应该如何进入这层 ARA。

## 阅读要点

- S2 的目标是从 S1 表示进入 fold stack、graph builder、probability container，再接近翻译结构。
- 历史 checkpoint 与过强 syntax-energy claim 已经被降级；当前重点是 graph assembly bottleneck 和概率容器。
- 任何 WMT 或 Transformer 替代 claim 都必须等待新证据，不允许靠旧 checkpoint 背书。
- 读取 evidence 时优先看 summary.json / trace.jsonl / README，再回到 logic/claims.md 更新 claim 状态。

## Reviewer 检查方式

- 证据目录里的数字是 claim 状态变化的依据。
- 如果 README 和 summary.json 冲突，以可复现 summary/trace 为准，并修正文档。

## 与英文原文的关系

- 英文原文保留为 canonical source，中文版本用于中文读者快速理解和审阅。
- 涉及实验数字、命令、文件路径、claim ID 时，应以英文原文和 evidence 目录中的机器输出为最终校验来源。
- 如果后续英文原文有 claim 状态更新，本文件也应同步更新。
