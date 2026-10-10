#!/usr/bin/env bash
# uv sync --locked --extra cuda --extra kinematics --extra export --extra viz
# Usage: bash scripts/a40.sh [no_physics|physics] [checkpoint-to-resume]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-no_physics}"
case "$TAG" in
  no_physics|physics) ;;
  *) echo "Usage: $0 [no_physics|physics] [checkpoint-to-resume]" >&2; exit 2 ;;
esac
NORN_RUN_DIR="${NORN_RUN_DIR:-outputs/gmt_a40/$TAG}"
NORN_RESUME=()
if [[ $# -gt 1 ]]; then NORN_RESUME=(--resume "$2"); fi
if [[ ! -f processed/gmt_dataset.npz ]]; then
  uv run --no-sync norn prepare --config configs/a40.json --output processed/gmt_dataset.npz
fi
PHYSICS=()
if [[ "$TAG" == physics ]]; then
  PRIORS="${NORN_PHYSICS_CONSTRAINTS:-processed/gmt_rigid_priors.npz}"
  if [[ ! -f "$PRIORS" ]]; then
    uv run --no-sync norn prepare-rigid-priors --config configs/a40.json \
      --dataset processed/gmt_dataset.npz --output "$PRIORS"
  fi
  PHYSICS=(--physics-constraints "$PRIORS")
fi
# no_physics never imports/builds/loads consistency priors or temporal smoothing.
uv run --no-sync norn train --config configs/a40.json --tag "$TAG" \
  --output "$NORN_RUN_DIR" "${PHYSICS[@]}" "${NORN_RESUME[@]}"
uv run --no-sync norn infer --checkpoint "$NORN_RUN_DIR/best.pt" --ages all \
  --output "$NORN_RUN_DIR/inference" --device cuda --formats npz dat netcdf
uv run --no-sync norn evaluate --checkpoint "$NORN_RUN_DIR/best.pt" \
  --dataset processed/gmt_dataset.npz --split validation --device cuda \
  --output "$NORN_RUN_DIR/validation.json"
uv run --no-sync norn animate --checkpoint "$NORN_RUN_DIR/best.pt" --ages all \
  --output "$NORN_RUN_DIR/evolution_0_60Ma.mp4" --device cuda --fps 6 --vmin 0 --vmax 80
