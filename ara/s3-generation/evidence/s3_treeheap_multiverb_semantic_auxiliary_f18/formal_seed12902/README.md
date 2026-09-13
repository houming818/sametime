# S3-TREEHEAP-MULTIVERB-SEMANTIC-AUXILIARY-F18

- Train/eval instances: `1344/336`
- Head AUROC: `{'warmup_root': 0.593568287037037, 'warmup_leaf': 0.5928509837962963, 'final_root': 0.6196712962962962, 'final_leaf': 0.6192297453703705}`
- WMT NLL delta: `-0.049341253715552646`
- F16 root deltas: `{'fold_root': 0.001180555555555518, 'read_facing_root': 0.0060763888888888395}`
- Gates: `{'O0_data_contract': True, 'O1_finite': True, 'O2_source_frozen': True, 'O2_model_updated': True, 'O3_checkpoint_reload': True, 'S0_wmt_nll_preserved': True, 'P1_aux_root_auroc': False, 'P2_aux_scale_consistency': True, 'P3_f16_fold_root_transfer': False, 'P4_f16_read_root_transfer': False}`
- Diagnosis: `semantic pressure insufficient at both scales; do not isolate FOLD`
