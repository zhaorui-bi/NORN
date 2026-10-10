#!/usr/bin/env bash
# Historical 0.4 Muller2019 GPML ablations; not the new GMT/tag experiment.
# Data-only experiments; never prepare/load physics priors or score test labels.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${NORN_PYTHON:-$PWD/.venv/bin/python}"
variants=("$@")
if ((${#variants[@]} == 0)); then
  variants=(control joint)
fi
for variant in "${variants[@]}"; do
  case "$variant" in
    control) config=configs/a40_ml_control.json ;;
    backbone) config=configs/a40_ml_backbone.json ;;
    loss) config=configs/a40_ml_loss.json ;;
    joint) config=configs/a40_ml_joint.json ;;
    *) echo "Unknown variant: $variant (control/backbone/loss/joint)" >&2; exit 2 ;;
  esac
  output="outputs/ml_${variant}_a40"
  mkdir -p "$output"
  state="fresh"
  if [[ -f "$output/last.pt" ]]; then
    state=$("$PYTHON" - "$config" "$output" <<'PY'
import json,sys
from pathlib import Path
from norn_earth.config import config_from_dict,load_config
root=Path(sys.argv[2])
old=json.loads((root/'run_manifest.json').read_text())['config']
if config_from_dict(old).to_dict() != load_config(sys.argv[1]).to_dict():
    raise ValueError('Existing run differs from this config; use a new directory')
summary=root/'training_summary.json'
complete=summary.is_file() and json.loads(summary.read_text())['completed']
print('complete' if complete and (root/'best.pt').is_file() else 'resume')
PY
)
  fi
  if [[ "$state" == "fresh" ]]; then
    "$PYTHON" -m norn_earth train --config "$config" > "$output/train.log" 2>&1
  elif [[ "$state" == "resume" ]]; then
    "$PYTHON" -m norn_earth train --config "$config" --resume "$output/last.pt" >> "$output/train.log" 2>&1
  else
    echo "$variant training already complete; verifying frozen validation score"
  fi
  "$PYTHON" -m norn_earth evaluate --checkpoint "$output/best.pt" \
    --dataset processed/dynamic_dataset.npz --split validation --device cuda --output "$output/validation.json" \
    > "$output/evaluation.log" 2>&1
  echo "$variant completed: $output/validation.json"
done
"$PYTHON" scripts/compare_ml_runs.py "${variants[@]}"
