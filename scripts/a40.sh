#!/usr/bin/env bash
# Run from the repository root after uv sync --locked --extra cuda --extra kinematics --extra export.
set -euo pipefail
# Optional first argument: checkpoint to resume. NORN_RUN_DIR selects a new run directory.
NORN_RUN_DIR="${NORN_RUN_DIR:-outputs/release_a40}"
NORN_RESUME=()
if [[ $# -gt 0 ]]; then NORN_RESUME=(--resume "$1"); fi
if [[ ! -f processed/release_dataset.npz ]]; then
  uv run --no-sync norn prepare --config configs/a40.json --output processed/release_dataset.npz
fi
if [[ ! -f processed/release_rigid_priors.npz ]]; then
  uv run --no-sync norn prepare-rigid-priors --config configs/a40.json \
    --dataset processed/release_dataset.npz --output processed/release_rigid_priors.npz
fi
uv run --no-sync norn train --config configs/a40.json --output "$NORN_RUN_DIR" "${NORN_RESUME[@]}"
uv run --no-sync norn infer --checkpoint "$NORN_RUN_DIR/best.pt" --ages all \
  --output "$NORN_RUN_DIR/inference" --device cuda --formats npz dat netcdf
uv run --no-sync norn evaluate --checkpoint "$NORN_RUN_DIR/best.pt" \
  --dataset processed/release_dataset.npz --split validation --device cuda \
  --output "$NORN_RUN_DIR/validation.json"
