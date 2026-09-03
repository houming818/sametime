# D12 predictive-state sketch evidence

Claim: `S3-PREDICTIVE-STATE-SKETCH-D12`

- `smoke_seed11201/`: taskd #356, paired 500-step GPU smoke on io.
- Both arms share initialization, data, seed, token CE, fixed sketch, and predictor.
- Only the predictive-to-TreeHeap gradient is detached in `probe-only`.
- Large checkpoint files remain on io and are intentionally excluded here.
- `smoke_seed11201/gradient_calibration.json`: taskd #358, eight-batch
  post-smoke CE/predictive gradient scale and cosine audit. Task #357 is the
  retained failed first attempt caused by a missing audit CLI argument.

The narrow gradient-channel mechanism is supported. The additional held-out
predictive MSE gain is only `0.0000306368`, so a meaningful representation or
generation-quality advantage remains open.
