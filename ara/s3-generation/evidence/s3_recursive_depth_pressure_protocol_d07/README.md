# D07 Pressure Protocol Training

This directory is reserved for evidence produced by the preregistered D07 run.

- Claim: `S3-RECURSIVE-DEPTH-PRESSURE-PROTOCOL-D07`
- Logic: `ara/s3-generation/logic/recursive_depth_pressure_protocol_training_d07.zh.md`
- Runner: `ara/s3-generation/scripts/run_recursive_depth_pressure_protocol.sh`
- Source: `ara/s3-generation/src/s3_recursive_depth_pressure_protocol_training.py`

The smoke run must finish before a formal run is scheduled. A failed P0, P1,
or P2 blocks automatic scaling.

## Result

Taskd `328` completed 120 steps. P0/P1/P3/P4 passed and P2 failed. Zeroing the
protocol improved NLL by 1.51-1.90, so the trainable language decoder bypassed
the protocol. Formal scaling was blocked.
