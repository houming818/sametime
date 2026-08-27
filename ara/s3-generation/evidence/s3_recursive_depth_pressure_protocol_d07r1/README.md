# D07R1 Frozen-Language Pressure Protocol

This run keeps the D07 rate-distortion design and freezes the inherited language
backbone so that only TreeHeap protocol parameters can learn.

- Claim: `S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07R1`
- Logic: `ara/s3-generation/logic/recursive_depth_pressure_protocol_training_d07r1.zh.md`
- Runner: `ara/s3-generation/scripts/run_recursive_depth_pressure_protocol_r1.sh`

## Result

Taskd `329` completed 300 steps. Shuffle damage increased to 0.59-0.62 NLL,
but zeroing the protocol still improved NLL by 1.36-1.44. The protocol carried
sample information at a harmful reference amplitude. Formal scaling was blocked.
