# S3-STONE2-CONTINUOUS-DEPTH-READ-C05 Evidence

This directory records the isolated C05 implementation and matched smoke runs.

- `impl_smoke/`: taskd 310, two-step implementation check.
- `formal_seed16501/`: taskd 311, 120-update matched smoke.
- Formal decision: `continuous_depth_read_not_supported`.
- G0, G1, G4, and G5 passed; G2 and G3 failed.
- No multi-seed or long training was authorized.

The experiment changes only the READ parameterization. It does not use the
quarantined IDEA-001 or any alternative hierarchical target.
