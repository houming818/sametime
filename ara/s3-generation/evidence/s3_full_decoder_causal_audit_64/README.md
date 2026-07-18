# S3 full decoder causal audit: 64-sample confirmation

Claim: `S3-FULL-DECODER-CAUSAL-C01`

Host: `io`, CPU only, frozen step-160K checkpoint, 64 held-out examples.

Conclusion: **source content is causal; current FOLD topology is not materially
used by the normal decoder path**.

| Gate | Result | Observation |
|---|---|---|
| P1 state causality | pass | wrong-sample state costs `+0.7448` to `+0.8084` NLL |
| P2 numeric permutation tolerance | fail | NLL invariant within `3.3e-7`; FP logit max `3.62e-4` exceeded the preregistered tolerance |
| P3 FOLD structure sensitivity | fail | root/middle sibling or half-swap damage at most `0.00834`, below `0.02` |

Artifacts:

- `command.sh`: exact invocation
- `stdout.log` and `stderr.log`: execution logs
- `summary.json`: checkpoint hash, configuration, all frontier metrics, gates

Boundary: frozen teacher-forced mechanism audit only. No quality, superiority,
semantic, world-model, or consciousness claim.
