# STONE-2 C04 grouped READ matched smoke

- Host/task: `io` taskd `309`
- Parent checkpoint SHA-256: `24d2b03c7a5f7441e3169ed40741544f62b1ce7db3e37ade7021558c226cb202`
- Stream SHA-256: `f64de353272c55d4ddfc0876c79cf749bd354fcb06f9c3e4a925999e623bdab3`
- Steps/batch: `120 / 16`
- Only READ kernel parameters were trainable.

All three arms were exactly function-equivalent at step zero. Resolution grouping
beat the fully shared READ kernel, but did not beat the equal-parameter interleaved
control:

```text
valid NLL shared / grouped / interleaved:
8.042080 / 8.030259 / 8.029222

test NLL shared / grouped / interleaved:
8.084835 / 8.071013 / 8.070220
```

Gate G3 failed. The result supports an untying candidate, not the proposed
coarse/middle/fine grouping. Formal training is not authorized.

`formal_seed16401/summary.json` is the valid result. `impl_smoke/` only checks
execution and exact initial equivalence.
