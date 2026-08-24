# README（中文读者版）

对应英文文件：`ara/README.md`  
所属主题：ARA 根目录

> 说明：这是递归生成并人工组织过的中文读者版。它保留原文件的 ARA 用途、claim/evidence 边界和 reviewer 读法；命令、ID、路径、指标名通常保留英文，方便和原文及实验输出对照。

## 文件用途

本文件说明当前目录如何组织、如何从 claim 找到 evidence，以及 reviewer 应该如何进入这层 ARA。

## 阅读要点

- ARA 的核心纪律是 claim 必须绑定 predict、experiment、evidence、decision 和 next step。
- Blog 负责叙事，ARA 负责审计；聊天记录不能替代证据文件。
- S3/STONE 生成航线的完整中间演化见
  [`s3-generation/EVOLUTION.zh.md`](s3-generation/EVOLUTION.zh.md)。该文件按技术问题
  组织成功、失败与勘误；`Cxx` 只作为 evidence 定位编号，不表示时间或能力等级。

## Reviewer 检查方式

- 维护本文时不要删除英文原文件；中文版本只作为 reader-facing mirror。

## 与英文原文的关系

- 英文原文保留为 canonical source，中文版本用于中文读者快速理解和审阅。
- 涉及实验数字、命令、文件路径、claim ID 时，应以英文原文和 evidence 目录中的机器输出为最终校验来源。
- 如果后续英文原文有 claim 状态更新，本文件也应同步更新。
