# F08 cross-language Echo controller r2

- Host: `io.grepcode.cn`
- taskd: `399`
- Seed: `11901`
- Runtime: `67.78 s`
- Model/theta: frozen
- Integrity gates: all passed
- Claim status: soft structured response supported; hard cross-language Echo not supported

Test positive coverage was `0.15778` for single native input, `0.16119` for an
RMS-matched uniform gain, `0.18194` for the shared filter controller, `0.31140`
for repeated native input, and `0.17720` for the test oracle. Shared filters did
not improve the hard token hit rate. The short atomic inputs expose only one or
two active internal-parent coordinates, so most topology filters degenerate.

See `summary.json` and `cases.json` for per-row coverage and decoded text.
