# Changelog

## 0.5.0

- Add `--tag no_physics|physics`: preserve backbone, data loss, split, seed, schedule and validation-MAE selection; isolate output directories and prohibit cross-tag resume.
- Make both public real-data modes consume same-age plate and deformation-network GMT snapshots, preserving exterior parts, holes and full metadata; never rotate exported coordinates twice.
- Numerically register complete feature sets, plate IDs, vertices and edge midpoints against original GPML at all 0–60 Ma anchors with a one-metre tolerance; reject mismatched model versions even with the same filename.
- Pin support files matching the GMTs' original Zahirovic2022/ANCHOR=0 frame; retain Muller2019 historical configurations/results without relabelling them as new GMT evidence.
- Retain zero-area incipient-network boundaries with explicit zero coverage and diagnostics; disable silent identity-rotation fallback for missing plate circuits.
- Bind derived physical priors to the frozen dataset and geometry source; no_physics does not import/load/instantiate consistency modules.
- Correct diagnostic spherical coverage for concave rings, antipodal ghosts, multiple parts, holes and finite minor-arc distance.
- Add manufactured paired-tag, frame/corruption, standalone-checkpoint and GMT regressions, plus a dedicated CPU/kinematics CI job.

## 0.4.0

- Make A40 and public reconstruction templates data-only ML baselines, explicitly forbidding physics artifacts/weights and temporal smoothing in this mode; omit physics objects, optimizer state and loss logging entirely.
- Add deterministic gated SFNO blocks with depthwise local mixing, SwiGLU, LayerScale, richer age Fourier features and optional input-reference residual regression.
- Add a per-record age-marginal Huber auxiliary data loss and explicit validation-MAE selection, with no modern/training-loss contamination of selection.
- Define matched control/backbone-only/loss-only/joint configurations and validation-only comparison tooling; preserve the historical physical configuration separately.
- Keep schema-2 legacy checkpoint inference and original explicitly constrained regression fixtures compatible; add pure-data gradient, masking, checkpoint/resume/export and guard tests.
- Promote the original pure SFNO control after validation: the new bundle did not improve MAE at the matched 500-step budget; retain the candidate and single-factor configurations rather than claiming an improvement.

## 0.3.0

- Bind one spatial feature field to each 0–60 Ma anchor in preparation, both training passes and standalone inference; stop broadcasting modern inputs across ages.
- Reconstruct continental masks and resolve plate/network geometry using one rotation model; add transported modern conditioning references, never historical labels.
- Freeze geometry ages, plate IDs, source/sidecar hashes and support diagnostics in schema-2 datasets/checkpoints; require retraining from legacy artifacts.
- Screen rigid-prior paths against age-matched boundary/network support at every physics node.
- Export matching geometry in NPZ/NetCDF and add MP4/GIF animation with moving continents, plate edges, fixed thickness colour limits and provenance.
- Add manufactured rotating-plate and dynamic two-pass/standalone/movie regression tests; update the core framework and A40 workflow.

## 0.2.1

- Stabilize spherical polygon orientation for repeated closing vertices and reversed rings.

- Manage CPU and CUDA 12.1 environments with uv, one committed lockfile and a development dependency group.
- Make CI tests independent of local datasets and verify Python 3.9/3.12, CPU SFNO, lint and built-wheel installation.
- Remove historical experiment scripts/configs, duplicate CLI wrappers, redundant requirements files and the unused SFNO compatibility module.
- Use the original framework figure and Methods documents; fix unsupported Markdown math macros.

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

Initial research prototypes, statistical baselines and numerical helpers. Historical scripts and archived README are available in Git history; they were removed from the current source tree in 0.2.1.
