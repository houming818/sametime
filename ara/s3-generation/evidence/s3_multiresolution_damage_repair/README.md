# Multiresolution Damage Repair

Complete input is FOLDed before addressed residual damage. The 34M-parameter
annealed base is frozen.

Decision: partial support. Parent-only repair recovered `72.38%` of leaf-state
MSE and restored affected token retrieval from `0.5573` to `0.9095`. The larger
root/sibling/path model recovered `69.94%`, and wrong addresses were cheap.
This supports parent-detail lifting redundancy, not address-conditioned repair.

Repair checkpoint:

`/mnt/nas/ara/s3-generation/evidence/s3_multiresolution_damage_repair/repair_kernels.pt`

See `summary.json`, `stdout.log`, and `checkpoints.sha256` for the complete
registered matrix and artifact identity.
