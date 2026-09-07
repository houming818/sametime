# F01 TreeHeap microscope formal result

- Task: nio-taskd 369
- Runtime: 111 seconds on the consumer RTX 3090
- Specimens: 20
- Depths: 5, 6, 7
- FOLD scales: 9
- K_up modes: native, bypass
- Result cells: 1080/1080
- Aggregate cells: 54/54
- Native-cell exact reproduction: 3/3 depths
- Native/bypass decoded-text changes: 538/540 paired cells

## Registered decisions

- P0 contract: pass
- P1 contiguous formed-text focus interval: pass at all three depths
- P2 endpoint instability: pass; `s=1` degrades strongly with depth
- P3 K_up causal focus contribution: pass; bypass moves the stable interval
- P4 shared versus adaptive focus: mixed; depth 5 is specimen-dependent, while
  depths 6 and 7 cluster around `s=0.707..0.8`

## Native formed-text intervals

| depth | contiguous passing scales |
|---:|---|
| 5 | 0.375, 0.5, 0.625, 0.707 |
| 6 | 0.25, 0.375, 0.5, 0.625, 0.707 |
| 7 | 0.125, 0.25, 0.375, 0.5, 0.625, 0.707 |

Operational formation is not translation correctness. The character-F2
diagnostic generally peaks closer to `0.707..0.8`, where output length is also
closer to instability. Full direct text remains in `results.json`.
