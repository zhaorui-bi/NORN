# Changelog

## 0.2.0

- Add an installable, configuration-driven `norn` CLI with preparation, training, evaluation, inference and a self-contained synthetic example.
- Correct periodic negative-longitude sampling, SHT latitude ordering, native/Gauss regridding and fractional-age interpolation.
- Reconstruct observation positions with pygplates at the actual age quadrature nodes.
- Use one spherical model and one observation operator for training, point predictions and exported maps.
- Integrate differentiable trajectory, moving-control-volume budget, ridge-birth and bounded regional-source constraints with explicit support masks and evidence IDs.
- Recompute and backpropagate anchor chunks immediately to bound activation memory; jointly optimize source coefficients.
- Save portable checkpoints with configuration, input features, optimizer/RNG state, data hashes and aligned metrics.
- Add grouped train/validation/test splits, independent test scoring, CPU CI, release packaging and regression tests.
- Archive v0.1 research scripts. Their checkpoints and reported challenge metrics are not compatible with the corrected pipeline.

## 0.1.0

Initial research prototypes, statistical baselines and numerical helpers. Preserved under `scripts/legacy` and `docs/archive` for provenance.
