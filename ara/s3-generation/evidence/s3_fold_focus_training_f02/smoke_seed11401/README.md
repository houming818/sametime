# F02 initial smoke failure

- nio-taskd: 370
- exit: 5 after all 30 requested steps
- failed gate: `step0_function`
- native/focus step-0 NLL delta: `3.11086436610708e-8`
- step-0 direct decoded text: exact
- all depth gradients: observed and finite
- frozen base, completion, bounded scale, reload: pass

The failure came from the round trip `sigmoid(logit(sqrt(0.5)))` in float32,
not from a decoded-text or structural mismatch. The P0 tolerance was not
relaxed. Smoke r1 uses an exact native-origin coordinate and a separate
evidence directory.
