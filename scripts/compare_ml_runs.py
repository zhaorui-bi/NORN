"""Compare completed, frozen data-only runs on validation, never on test."""

import argparse
import json
from pathlib import Path

from norn_earth.utils.hashing import sha256_file, write_manifest

VARIANTS = {"control", "backbone", "loss", "joint"}


def compare(variants, root=Path("outputs")):
    runs = {}
    common = None
    for name in variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant: {name}")
        directory = root / f"ml_{name}_a40"
        manifest = json.loads((directory / "run_manifest.json").read_text())
        summary = json.loads((directory / "training_summary.json").read_text())
        evaluation = json.loads((directory / "validation.json").read_text())
        c = manifest["config"]
        training = c["training"]
        physics = c["physics"]
        if (
            training["objective"] != "data_only"
            or training["temporal_smooth_weight"] != 0
            or physics["constraints"] is not None
            or any(value != 0 for key, value in physics.items() if key.endswith("weight"))
        ):
            raise ValueError(f"{name} is not a pure-data baseline")
        if "physics" in manifest["provenance"] or not summary["completed"]:
            raise ValueError(f"{name} has unexpected factors or is incomplete")
        if evaluation["split"] != "validation" or training["selection_metric"] != "validation_mae":
            raise ValueError("Comparison/selection must use validation MAE only")
        if evaluation["checkpoint_sha256"] != sha256_file(directory / "best.pt"):
            raise ValueError("Scored checkpoint has changed")
        settings = {
            "dataset_sha256": manifest["provenance"]["dataset_sha256"],
            "grid": c["grid"],
            "max_age_ma": c["max_age_ma"],
            "anchor_step_myr": c["anchor_step_myr"],
            "data": c["data"],
            "training": {
                key: value for key, value in training.items() if key != "observation_huber_weight"
            },
        }
        if common is not None and common != settings:
            raise ValueError("Runs differ in data, split, age domain, optimizer or step budget")
        common = settings
        mae = evaluation["metrics"]["age_marginal_mean_mae_km"]
        if abs(mae - summary["best_validation_score"]) > 1e-5:
            raise ValueError("Best checkpoint and validation selection score disagree")
        runs[name] = {
            "directory": str(directory),
            "model": c["model"],
            "observation_huber_weight": training["observation_huber_weight"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "validation": evaluation["metrics"],
            "parameters": summary["runtime"]["parameters"],
            "completed_steps": summary["completed_steps"],
            "elapsed_s": summary["elapsed_s"],
            "peak_cuda_memory_mb": summary["peak_cuda_memory_mb"],
            "final_modern_area_rmse_km": summary["final"]["modern_area_rmse_km"],
        }
    if not runs:
        raise ValueError("At least one completed variant is required")
    selected = min(runs, key=lambda name: runs[name]["validation"]["age_marginal_mean_mae_km"])
    result = {
        "protocol": "pure ML; same data/split, seed and optimizer-step budget; validation-only selection",
        "runs": runs,
        "selected_by_validation_mae": selected,
        "test_rescored": False,
        "shared_settings": common,
        "limitation": "One seed; equal optimizer steps, not equal FLOPs. Joint vs control does not isolate backbone from loss; run both single-factor variants for attribution.",
    }
    if "control" in runs:
        baseline = runs["control"]["validation"]["age_marginal_mean_mae_km"]
        result["relative_mae_improvement_vs_control"] = {
            name: 1 - run["validation"]["age_marginal_mean_mae_km"] / baseline
            for name, run in runs.items()
        }
    write_manifest(root / "ml_baselines" / "comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", choices=sorted(VARIANTS))
    args = parser.parse_args()
    print(json.dumps(compare(args.variants), ensure_ascii=False, indent=2))
