# D12-R1 evidence

- Claim: `S3-PREDICTIVE-STATE-SKETCH-D12-R1`
- Host: `io.grepcode.cn`, RTX 3090, 270W limit
- taskd: `359`
- Command: `bash ara/s3-generation/scripts/run_predictive_state_sketch_d12_r1.sh`
- Budget: 10,000 steps per arm, batch 16
- Data: `NioClean-ZHEN-S098-7M-v2`
- Decision: narrow mechanism supported; meaningful representation gain remains open

Both arms completed 10,000 steps at cursor 160301 and processed 3,806,969 target tokens.
Predictive gradient changed Test NLL by -0.000967 and fixed-sketch MSE by -0.000476
(-0.0468% relative); retrieval@1 remained 0.127. All safety, frozen-source, receipt,
and reload gates passed.

Large `checkpoint_latest.pt` files remain on the remote host and are intentionally excluded
from this Git evidence copy. `comparison.json` is the compact paired result; arm summaries,
traces, and 1K progress receipts provide the audit trail.
