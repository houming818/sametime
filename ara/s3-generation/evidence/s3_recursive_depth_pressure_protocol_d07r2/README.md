# D07R2 Bounded Protocol Gain

D07R2 freezes both inherited backbones and learns one shared bounded gain before
the pressure protocol slots enter recursive FOLD.

- Claim: `S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R2`
- Logic: `ara/s3-generation/logic/recursive_depth_pressure_protocol_training_d07r2.zh.md`
- Runner: `ara/s3-generation/scripts/run_recursive_depth_pressure_protocol_r2.sh`

## Result

Taskd `330` completed 600 steps. Native beat both shuffled and zero protocols at
all depths, but zero damage was only 0.036-0.071 NLL and missed the preregistered
0.10 causal gate. The result is a weak, consistently signed input signal, not a
supported private protocol. Formal scaling was blocked.
