# F07 structured filter bank smoke

- Host: `io.grepcode.cn`
- taskd: `397`
- Seed: `11801`
- Runtime: `38.36 s`
- Checkpoint: frozen TreeHeap-106M E01 checkpoint
- Theta: frozen F04 theta
- Result: all integrity gates passed
- Claim status: structured response anisotropy supported; semantic recovery open

The strongest base-channel route response at `epsilon=+0.1` came from
`alternating-highpass` (`route_max_js=0.0008857`, 11 branch flips), despite a
smaller FOLD displacement than the smooth energy filters. No filter recovered
the requested Sisyphus/push translation. See `summary.json`,
`filter_interventions.json`, and `analytic_filter_bank.csv` for the complete
evidence.
