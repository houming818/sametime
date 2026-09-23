# S1 Probability-Residual Tree A12

A12 fixes the fit/dev split and varies only the Monte Carlo search seed across
a preregistered depth ladder. Tasks `563-574` ran 12 formal arms on the RTX
3090: depths `3/4/5/6`, three search seeds per depth, and 512 proposals per arm.

All four depths met the replication gates. Every formal arm improved dev NLL,
improved sealed-test NLL relative to its deterministic initial tree, beat its
random-route test control, preserved probability mass, closed its path
residual identity, and retained at least 96.875% leaf utilization.

The depth trend is not monotonic. Depth 3 has the strongest median test gain;
depth 6 has the weakest. This result supports reproducible F search but does
not identify the best semantic capacity. A frozen downstream comparison is
needed before selecting a production embedding depth.

Evidence layout:

- `aggregate_summary.json`: preregistered replication decision by depth
- `formal/depth_*/seed_*/summary.json`: per-arm metrics and gates
- `formal/depth_*/seed_*/trace.jsonl`: all 512 proposals per arm
- `taskd_logs/563.log` through `taskd_logs/574.log`: scheduler logs
- `smoke/`: interface smoke only; excluded from the formal claim
