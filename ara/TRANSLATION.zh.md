# ARA 递归中文镜像说明

本目录为 ARA Markdown 文档提供递归中文读者版。

## 生成规则

- 英文原文保留不动。
- 对每个 `.md` 文件，生成同路径同名的 `.zh.md` 中文读者版。
- 已有人写中文版本的文件不会覆盖，例如 `ara/PAPER.zh.md`。
- 中文版本保留 claim ID、路径、命令、指标名、状态词等英文符号，方便和 evidence 机器输出对照。
- 若英文原文和中文版本冲突，以英文原文及 evidence 目录中的 `summary.json` / `trace.jsonl` 为准。

## 已覆盖文件

- `ara/PAPER.zh.md`
- `ara/PUBLIC.zh.md`
- `ara/README.zh.md`
- `ara/m0-treeheap-math/evidence/README.zh.md`
- `ara/m0-treeheap-math/evidence/algebraic_decoder_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/deductive_inductive_kernel_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/kernel_convolution_ops_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/primitive_plus_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/soft_plus_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/soft_plus_probe/glm_audit_summary.zh.md`
- `ara/m0-treeheap-math/evidence/structural_c05_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/trainability_quiz/README.zh.md`
- `ara/m0-treeheap-math/evidence/treeheap_diff_algebra_probe/README.zh.md`
- `ara/m0-treeheap-math/evidence/treeheap_math_probe/README.zh.md`
- `ara/m0-treeheap-math/logic/claims.zh.md`
- `ara/m0-treeheap-math/logic/experiments.zh.md`
- `ara/m0-treeheap-math/logic/predicts.zh.md`
- `ara/m0-treeheap-math/logic/problem.zh.md`
- `ara/m0-treeheap-math/logic/soft_treeheap.zh.md`
- `ara/m0-treeheap-math/logic/solution/algebra.zh.md`
- `ara/s1-echo/evidence/README.zh.md`
- `ara/s1-echo/evidence/s1_corpus_embedding_kernel_probe/README.zh.md`
- `ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe/README.zh.md`
- `ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe_b32/README.zh.md`
- `ara/s1-echo/evidence/s1_wmt_echo_kernel_probe/README.zh.md`
- `ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe/README.zh.md`
- `ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe_common512/README.zh.md`
- `ara/s1-echo/evidence/s1_world_model_compound_probe/README.zh.md`
- `ara/s1-echo/evidence/shallow_treeheap_s1_probe/README.zh.md`
- `ara/s1-echo/logic/claims.zh.md`
- `ara/s1-echo/logic/experiments.zh.md`
- `ara/s1-echo/logic/problem.zh.md`
- `ara/s1-echo/src/environment.zh.md`
- `ara/s2-translation/evidence/README.zh.md`
- `ara/s2-translation/evidence/frame_probe_2h_queue/output/README.zh.md`
- `ara/s2-translation/logic/claims.zh.md`
- `ara/s2-translation/logic/experiments.zh.md`
- `ara/s2-translation/logic/predicts.zh.md`
- `ara/s2-translation/logic/problem.zh.md`
- `ara/s2-translation/logic/solution/architecture.zh.md`
- `ara/s2-translation/logic/solution/theory.zh.md`
- `ara/s2-translation/logic/solution/treeheap_algebra.zh.md`
- `ara/s2-translation/src/environment.zh.md`
- `ara/s3-generation/logic/problem.zh.md`

## 维护规则

当新增或修改 ARA Markdown 时，应同步更新对应 `.zh.md`。如果 claim 状态发生变化，应优先更新英文 canonical source，再更新中文读者版。
