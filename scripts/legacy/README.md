# Historical research scripts

These files preserve the v0.1 experiments. They are **not supported entry points** and their relative paths may no longer resolve after archiving.

The A40 prototypes sampled negative longitudes incorrectly, used native grids as Legendre–Gauss inputs without conversion, and omitted paleo-position reconstruction. One hybrid experiment also used a local correction during training that was absent from exported maps. Historical challenge scores must not be treated as validation of v0.2.

Use `norn prepare`, `norn train`, `norn infer` and `norn evaluate`. Old checkpoints cannot be loaded by the v0.2 checkpoint schema; retraining is required.
