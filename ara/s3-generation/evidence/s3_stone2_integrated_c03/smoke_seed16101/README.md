# STONE-2 C03 integrated smoke

- Mode: `smoke`
- Claim: `S3-STONE2-INTEGRATED-C03`
- io task: `292`
- Result: `8/9` preregistered structural gates passed; `S7_multilevel_positive` failed.
- Formal training authorized: **no**.

The pretrain and both matched task arms executed with finite gradients. FOLD/UNFOLD
closure, no-STOP structure, parameter gradients, source shuffle, Butterfly identity,
pair break, non-fixed generation and exact reload all passed. Disabling all multilevel
READ updates increased Test NLL by `0.13106698`, but removing any one effective depth
reduced NLL by about `0.0027` to `0.0043`; therefore the original per-depth causal gate
did not pass.

Frozen follow-up task `293` enumerated coarse/middle/fine group subsets. All three
groups had positive Shapley contributions (`0.0543`, `0.0538`, `0.0229`), while all
pair interactions were negative. This supports distributed multiresolution information
with redundancy/interference, not independent per-depth benefit and not a retroactive
pass of S7. See `depth_interaction_audit.json` and the C03-D01 logic note.

Smoke validates executable data flow and narrows the architecture problem. It is not
product evidence.
