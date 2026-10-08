#!/usr/bin/env bash
set -euo pipefail
uv run --no-sync norn demo --output outputs/demo --device "${NORN_DEVICE:-auto}"
uv run --no-sync norn train --config outputs/demo/config.json
uv run --no-sync norn infer --checkpoint outputs/demo/run/best.pt --ages 0 13.4 30 60 \
  --output outputs/demo/predictions --formats npz dat
