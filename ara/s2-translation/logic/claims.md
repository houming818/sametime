# S2 Claims Registry

> Auto-generated from 15 S2 blog posts + experimental evidence.
> Each claim has: status, evidence pointer, source blog.

---

## 1. 语义→结构可预测

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-001 | 语义向量编码折叠动作类型信息 | ✅ 验证 | Phase A: PCA 128D→MLP, AUC=0.64 | S2-15 |
| C-002 | 32D 语义空间已足够预测结构 | ✅ 验证 | 32D Top-1=95% ≈ 128D Top-1=95% | S2-15 |
| C-003 | EN→ZH 跨语言结构预测可行 | ✅ 验证 | D1: ZH语义→EN折叠, AUC=0.70 | S2-15 |
| C-004 | ZH→EN 跨语言结构预测可行 | ✅ 验证 | E2a: EN语义→ZH折叠, AUC=0.67 | S2-15 |
| C-005 | 折叠动作类型可被小 MLP 预测 | △ 部分 | Top-5=96% (VP), Top-5=96% (NP) | S2-15 |

## 2. 短语坍缩

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-006 | PP 短语模式极度坍缩 (~3) | ✅ 验证 | grammar_atlas_v4: 207 patterns, 96.6% top-5 | S2-14 |
| C-007 | VP 短语模式高度坍缩 (~44) | ✅ 验证 | 973 patterns, 91.2% top-50 | S2-14 |
| C-008 | NP 短语模式中等坍缩 (~150) | ✅ 验证 | 7300 patterns, 81.8% top-50 | S2-14 |
| C-009 | 子句层（SUB）不坍缩 | ✅ 验证 | 7825 patterns, 无 clear collapse | S2-14 |
| C-010 | 中文 VP 坍缩弱于英文 (8775 vs 973) | ✅ 验证 | fold_zh.py: 50K zh sentences | S2-15 |

## 3. 折叠发现

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-011 | 头检测可行 (Head F1=87.6%) | ✅ 验证 | D2a: token-level MLP classification | S2-15 |
| C-012 | 跨度检测可行 (Span F1=88.2%) | ✅ 验证 | D2a: span membership classifier | S2-15 |
| C-013 | 动作分类可行 (Action Top-5=96.7%) | ✅ 验证 | D2a: given gold head, predict action | S2-15 |

## 4. 折叠自编码

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-014 | 折叠表示可无损重建句子 | ✅ 验证 | autoencoder: BLEU=98.4% (oracle edges) | S2-15 |
| C-015 | 模板边分配仍有合理 BLEU | △ 部分 | BLEU=98.4% (template edges, UAS=50%) | S2-14 |

## 5. Graph Assembly

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-016 | 最近节点是最优简单启发式 | ✅ 验证 | Graph bench: 56.5% UAS, beats template+MLP+beam | S2-15 |
| C-017 | 100% 的 UAS 差距来自子节点分配 | ✅ 验证 | D3: Oracle ablation, Gold Child→+41% | S2-15 |
| C-018 | 模板规则打败所有学习预测器 | ✅ 验证 | 55.2% (template) > 48% (MLP) > 47% (LR) | S2-15 |
| C-019 | PP 附着歧义占错误的 35% | ✅ 验证 | Error atlas: prep errors dominate | S2-15 |
| C-020 | 97.5% 的错误中正确父节点比最近更远 | ✅ 验证 | Non-nearest distance analysis | S2-15 |

## 6. 概率容器

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-021 | 每个子节点可携带父节点概率分布 | ✅ PoC | P4: ProbFoldNode, 3 modes tested | S2-15 |
| C-022 | 非交换拼接可区分排列但能量未对齐句法 | ❌ 未验证 | path_tensor: energy range 0.06 but gold not min | latest |
| C-023 | TreeHeap 路径编码 token ID 而非语法角色 | ✅ 验证 | Path analysis: cat&look share cluster | latest |
| C-026 | 小固定槽位预算可以覆盖绝大多数 FoldNode | ✅ 验证 | strategy_audit: degree<=4 covers 99.0%, degree<=5 covers 99.8% | latest |
| C-027 | 当前 3-epoch TreeHeap 128D 向量过度坍缩，不适合直接做 syntax energy | ✅ 验证 | strategy_audit: TreeHeap off-diagonal cosine mean 0.9849; tensor margin near 0 | latest |
| C-028 | Parent 概率容器的 top-k 覆盖足够高，支持延迟坍缩 | ✅ 验证 | strategy_audit: gold parent top1=93.1%, top3=99.9%, top5=100% | latest |

## 7. 翻译

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| C-024 | 词对词翻译中英不可行 | ✅ 验证 | mt_v1: BLEU≈0 (OOV 88%), v0 BLEU=24.6 was bug | S2-15 |
| C-025 | 折叠词典存在 (ZH→EN pattern mapping) | ✅ PoC | fold_lexicon: NP 58%, VP 47% deterministic | S2-15 |

## 8. 未验证/开放问题

| Claim ID | Claim | Status | Evidence | Source |
|----------|-------|--------|----------|--------|
| O-001 | 能量最小化能否泛化到 N>3 token? | ❓ 未验证 | 3-token ✓, >3 未知 | latest |
| O-002 | 概率容器能否在翻译/生成阶段消歧? | ❓ 未验证 | 仅 Graph Builder 阶段测试 | S2-15 |
| O-003 | TreeHeap 更多 epoch 是否改善角色区分? | ❓ 未验证 | 3-epoch vs N-epoch 对比未做 | latest |
| O-004 | 非交换张量积是否可行（128^n 维度爆炸）? | ❓ 部分 | 3-token ✓, >3 需投影近似 | latest |
| O-005 | Role-slot filling 是否能替代 parent edge prediction? | ❓ 未验证 | degree/pattern collapse 强，但 slot model 尚未与 edge model 对比 | latest |

---

## Evidence Bindings

| Evidence File | Claims |
|---------------|--------|
| `phase_a_results.json` | C-001, C-002 |
| `phase_d1_results.json` | C-003 |
| `e2a_results.json` | C-004 |
| `grammar_atlas_v4_results.json` | C-006, C-007, C-008, C-009 |
| `fold_zh_results.json` | C-010 |
| `phase_d3_results.json` | C-017 |
| `p1_residual.py` output | C-016, C-018 |
| `diagnose_gap.py` output | C-019, C-020 |
| `autoencoder.py` output | C-014, C-015 |
| `mt_v1.py` output | C-024 |
| `fold_lexicon.json` | C-025 |
| `strategy_audit/strategy_audit_summary.json` | C-026, C-027, C-028 |
