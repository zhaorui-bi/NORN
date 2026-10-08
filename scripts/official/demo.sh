#!/usr/bin/env bash
set -euo pipefail
NORN_PYTHON="${NORN_PYTHON:-.venv/bin/python}"
"$NORN_PYTHON" -m norn_earth demo --output outputs/demo --device "${NORN_DEVICE:-auto}"
"$NORN_PYTHON" -m norn_earth train --config outputs/demo/config.json
"$NORN_PYTHON" -m norn_earth infer --checkpoint outputs/demo/run/best.pt --ages 0 13.4 30 60 \
  --output outputs/demo/predictions --formats npz dat
