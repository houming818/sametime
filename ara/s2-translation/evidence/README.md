# Evidence Directory

Organized by experiment phase. Each result file supports specific claims in `/ara/logic/claims.md`.

## Results Files

### Phase A: Semantic → Fold Action
| File | Content | Supports |
|------|---------|----------|
| `phase_a_results.json` | Top-1/5 at 32/64/128/256 dims | C-001, C-002 |

### Phase D1: Cross-lingual
| File | Content | Supports |
|------|---------|----------|
| `phase_d1_results.json` | ZH→EN AUC=0.701 | C-003 |
| `e2a_results.json` | EN→ZH AUC=0.671 | C-004 |

### Phrase Analysis
| File | Content | Supports |
|------|---------|----------|
| `grammar_atlas_v4_results.json` | PP/VP/NP coverage curves (EN) | C-006-009 |
| `fold_zh_results.json` | PP/VP/NP coverage curves (ZH) | C-010 |
| `top_pos_skeletons.json` | Top 1000 POS patterns (1M sentences) | C-006 |

### Fold Lexicon
| File | Content | Supports |
|------|---------|----------|
| `fold_lexicon.json` | ZH→EN pattern mapping (NP 58%, VP 47% deterministic) | C-025 |

### Structure Binding
| File | Content | Supports |
|------|---------|----------|
| `phase_d2_results.json` | Oracle=97.4 BLEU, Predicted=97.4 BLEU | C-014-015 |

### Oracle Ablation
| File | Content | Supports |
|------|---------|----------|
| `phase_d3_results.json` | Head/Span/Child oracle UAS values | C-017 |

### Edge Predictor / Graph Assembly
| File | Content | Supports |
|------|---------|----------|
| `phase_e1_results.json` | Learned edge predictor UAS=48% | C-018 |
| `phase_e1b_results.json` | MLP+template prior UAS=48% | C-018 |
| `role_slots_results.json` | Degree distribution, slot patterns | C-007-008 |

### Translation
| File | Content | Supports |
|------|---------|----------|
| (console) `mt_v1.py` output | BLEU=0.5, OOV=32% | C-024 |

## Log Files
| File | Content |
|------|---------|
| `local_train.log` | Local training runs |
| `q_run.log` | Remote GPU queue manager |
| `nio_thinking.log` | Thinking/decision traces |

## Experimental Data Sizes
| Experiment | Sentences | Tokens/Pairs |
|-----------|-----------|-------------|
| grammar_atlas_v4 | 100,000 | 359K NP + 86.5K VP + 158K PP |
| fold_zh.py | 50,000 | 161K NP + 110K VP |
| phase_a.py | 10,000 | 792K fold actions |
| Phase D2a | 10,000 | 180K tokens |
| Phase D3 | 3,000 | 75K fold nodes |
| Phase E1 (RF training) | 10,000 | 140K (parent,child) pairs |
| diagnose_gap.py | 5,000 | 24.7K fold nodes |
