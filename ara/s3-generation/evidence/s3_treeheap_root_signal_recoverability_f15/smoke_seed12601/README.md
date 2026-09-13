# S3-TREEHEAP-ROOT-SIGNAL-RECOVERABILITY-F15

- Cases: `256`; train/test: `192/64`
- Gates: `{'O0_exhaustive_position_holdout': True, 'O1_finite': True, 'O2_frozen_checkpoint': True, 'C0_shuffled_control_rejected': True, 'P1_native_fold_root_recoverable': True, 'P2_native_read_root_recoverable': True}`
- Diagnosis: `source count reaches read-facing root; F14 failure is access/alignment/scale, not erasure`
- Runtime seconds: `10.89`

This frozen synthetic probe tests source-count recoverability, not translation quality.
