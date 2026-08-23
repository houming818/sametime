# STONE-2 核心数据固化记录

日期：2026-08-23

任务：io `294`

结果：完成

## 输入 evidence

- 双语：`NioScore-ZHEN-14M-v1`，14,170,275 对影子评分记录；
- 中文 QA：`NioScore-ZH-QA-8451K-v1`，8,451,252 条关系评分记录；
- 中文自然文本：`NioAudit-ZH-Text-2985K-v1`，2,984,702 条完整性记录。

评分与 flags 均是影子 metadata；源语料没有被覆盖或删除。

## 输出 release

| Release | 规则 | 行数 | SHA-256 前缀 |
|---|---|---:|---|
| `NioText-ZH-Integrity-2985K-v1` | 排除 mojibake、extreme repetition | 2,972,976 | `04a90d88` |
| `NioClean-ZHEN-S098-7M-v2` | relation score >= 0.98 | 7,304,358 | `29913486` |
| `NioQA-ZH-S090-v1` | relation score >= 0.90 | 4,767,788 | `7a34840d` |
| `NioQA-ZH-S095-v1` | relation score >= 0.95 | 4,180,947 | `cffb804a` |
| `NioQA-ZH-S098-v1` | relation score >= 0.98 | 3,375,921 | `469b5ff1` |

数据文件位于 io 的 `/home/nio/datasets/nio/releases/`。Git 只保存 manifest、
来源、行数和哈希，不复制约 19 GB 的训练正文。

## 审计

物化器逐分片验证源压缩文件 SHA 和行数，要求分片连续；输出采用临时文件、
`fsync` 与原子替换。任务完成后又独立运行 `sha256sum` 读取五个完整大文件，
结果与 manifest 全部一致。五个 manifest 的哈希和统一 root hash 也在本地复算
通过。

STONE-2 核心数据根哈希：

`75caafdc24058eb96a957fd680b41789843eb3726e4febb4a110b7c96b38be29`

该记录只证明数据身份与筛选合同闭合，不证明筛选阈值必然提升模型质量。阈值
收益仍需同初始化、同 token/update 预算的训练对照验证。
