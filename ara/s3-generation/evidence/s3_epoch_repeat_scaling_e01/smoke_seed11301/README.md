# Epoch Repeat Scaling real-checkpoint smoke

Claim: `S3-EPOCH-REPEAT-SCALING-E01`

Remote host: `io.grepcode.cn`

taskd: `361`

Both D11 arms resumed their model and AdamW optimizer state at
`step=25,000/cursor=400,488`, trained 20 steps with batch 64, and reached
`step=25,020/cursor=401,768`. All finite-state, frozen-source, update, cursor,
bound and reload gates passed. The 106M arm completed without OOM or CUDA
failure.

Large checkpoint files remain on `io` and are intentionally excluded from this
evidence copy.
