# STONE-2 checkpoint energy-gradient audit

- Final io task: `301`
- Frozen checkpoint: C03 smoke seed 16101 pretrain best
- Data: 224 real validation windows, 32 each at widths 4..256
- Dtype: float64 FOLD/Jacobian audit
- Decision: `do_not_train_energy_carrier_yet`

No sibling pair had cancellation ratio below `0.10`; the global minimum was
`0.58384`. The artificial alternating-sign singularity therefore lacked an
existence signal in this checkpoint. The carrier remained numerically closed,
but its median root-to-leaf gradient fell from `0.03131` at width 4 to `0.00370`
at width 256, while the current formula fell from `0.04459` to `0.01483`.

This evidence stops the proposed training ablation. It does not reject energy
carriers universally and does not make a language-quality or S7 claim.
