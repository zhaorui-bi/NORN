#!/usr/bin/env bash
# Run from the repository root after installing the tested A40 environment.
set -euo pipefail
NORN_PYTHON="${NORN_PYTHON:-.venv/bin/python}"
"$NORN_PYTHON" -m norn_earth prepare --config configs/a40.json --output processed/release_dataset.npz
"$NORN_PYTHON" -m norn_earth prepare-rigid-priors --config configs/a40.json \
  --dataset processed/release_dataset.npz --output processed/release_rigid_priors.npz
"$NORN_PYTHON" -m norn_earth train --config configs/a40.json "$@"
"$NORN_PYTHON" -m norn_earth infer --checkpoint outputs/release_a40/best.pt --ages all \
  --output outputs/release_a40/inference --device cuda --formats npz dat netcdf
"$NORN_PYTHON" -m norn_earth evaluate --checkpoint outputs/release_a40/best.pt \
  --dataset processed/release_dataset.npz --split validation --device cuda \
  --output outputs/release_a40/validation.json
