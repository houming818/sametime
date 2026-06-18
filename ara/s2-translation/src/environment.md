# Environment & Dependencies

## Hardware
- **Remote GPU server**: io.grepcode.cn
- **GPU**: NVIDIA 3090 (24GB)
- **CPU**: Varies (i5-4690 local, Xeon remote)
- **Disk**: /mnt/nas (shared storage)

## Python Environment
```bash
python=3.10
torch>=2.0 (inference only)
sentence-transformers==3.x
spacy==3.8.x
  en_core_web_sm, en_core_web_md, zh_core_web_sm
jieba==0.42
sentencepiece==0.2.1
scikit-learn==1.x
numpy, matplotlib
```

## Key Models
| Model | Path | Purpose |
|-------|------|---------|
| TreeHeap (L0+L1) | `/mnt/nas/datasets/wmt_massive/checkpoints/anchor_tree_massive_ep3.pt` | 128D semantic/syntactic vectors |
| SentencePiece BPE | `/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model` | Tokenization |
| Sentence Transformer | `all-MiniLM-L6-v2` (384D) | Semantic embeddings (proxy for L1) |

## Dataset
- **WMT Massive ZH-EN**: `/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv` (2.5GB, ~14M pairs)
- **SentencePiece model**: `/mnt/nas/datasets/wmt_massive/sp_bpe_massive.model`
- **Checkpoints**: `/mnt/nas/datasets/wmt_massive/checkpoints/`

## Key Scripts (Phase→ARA mapping)
| Experiment | Script | Evidence |
|-----------|--------|----------|
| Phase A (semantic→fold) | `phase_a.py` | `phase_a_results.json` |
| Phase D1 (cross-lingual) | `phase_d1.py` | `phase_d1_results.json` |
| Phase D2a (structure binding) | `phase_d2a.py` | console output |
| Phase D2bc (end-to-end UAS) | `phase_d2bc.py` | `phase_d2_results.json` |
| Phase D3 (oracle ablation) | `phase_d3.py` | `phase_d3_results.json` |
| Phase E1 (edge predictor) | `phase_e1.py` | console |
| Phase E1b (MLP+prior) | `phase_e1b.py` | `phase_e1b_results.json` |
| P1 (residual classifier) | `p1_residual.py` | console |
| P2 (beam search) | `p2_beam.py` | console |
| P3 (probability container) | `p3_container.py` | console |
| P4 (probability demo) | `p4_probability_role.py` | console |
| Fold analysis (EN) | `grammar_atlas_v4.py` | `grammar_atlas_v4_results.json` |
| Fold analysis (ZH) | `fold_zh.py` | `fold_zh_results.json` |
| Fold lexicon | `fold_lexicon.py` | `fold_lexicon.json` |
| Autoencoder | `autoencoder.py` | console |
| MT v0/v1 | `mt_v0.py`, `mt_v1.py` | console |
| Error atlas | `diagnose_gap.py` | console |
| Tensor PoC | `path_tensor.py`, `tensor_poc.py` | console |
| Geometric graph | `graph_geo.py` | console |
| 128D extraction | `extract_128d.py` | console |

## Random Seeds
- All experiments: `np.random.RandomState(42)`
- Train/test split: fixed 80/20

## Execution Order (to reproduce)
1. `grammar_atlas_v4.py` → phrase structure statistics
2. `phase_a.py` → semantic→fold action predictability
3. `phase_d2a.py` → structure binding
4. `phase_d3.py` → oracle ablation
5. Graph builder methods: `graph_bench.py`
6. Residual/predictor: `p1_residual.py` → `p2_beam.py` → `p3_container.py`
7. Tensor experiments: `tensor_poc.py` → `path_tensor.py`
