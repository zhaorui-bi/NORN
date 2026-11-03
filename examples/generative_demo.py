"""Reproducible synthetic comparison and generative pipeline, CPU by default.

Run: uv run --no-sync python examples/generative_demo.py --output outputs/generative_demo
This is manufactured software validation, not a geological accuracy benchmark.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from norn_earth.config import load_config
from norn_earth.data.demo import create_demo, truth
from norn_earth.data.preparation import load_dataset, prepare_dataset
from norn_earth.evaluation.missing import missing_field_metrics, training_query_support
from norn_earth.geometry.regrid import gauss_area_weights_km2, gauss_grid
from norn_earth.inference import NornPredictor, export_prediction
from norn_earth.losses.objective import ReconstructionObjective
from norn_earth.models.diffusion import train_latent_prior
from norn_earth.models.ensemble import encode_ensemble
from norn_earth.models.posterior import evaluate_posterior, sample_posterior
from norn_earth.training.trainer import train
from norn_earth.utils.hashing import write_manifest


def run(output, steps=100, prior_steps=200, encoding_steps=100, device="cpu"):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    path = create_demo(
        output / "data", device, nlat=8, nlon=16, width=8, blocks=1, steps=steps, anchor_step=10
    )
    common = load_config(path).with_tag("no_physics")
    common.model.head_mode = "input_residual"
    common.model.spatial_rank = 4
    common.model.context_rank = 2
    common.training.selection_metric = "validation_mae"
    common.training.num_threads = 2
    common.data.test_age_window_ma = [25, 35]
    common.training.validation_every = 10
    common.training.learning_rate = 0.003
    dataset = prepare_dataset(common, output / "dataset.npz")
    common.data.prepared = str(dataset)
    data = load_dataset(dataset, common)
    objective = ReconstructionObjective(data, common, device)
    lat, lon, _ = gauss_grid(common.grid.nlat, common.grid.nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat[::-1])
    true_fields = np.stack([truth(lon_grid, lat_grid, a) for a in common.anchor_ages])
    support = training_query_support(objective)
    areas = gauss_area_weights_km2(common.grid.nlat, common.grid.nlon)[::-1]
    np.savez_compressed(
        output / "comparison_support.npz",
        training_support=support,
        age_ma=common.anchor_ages,
        latitude=lat[::-1],
        longitude=lon,
    )
    results = {}

    def score(fields):
        with torch.no_grad():
            metrics = {}
            for split in ("validation", "test"):
                entry = objective.observation_metrics(fields, split)
                metrics[split] = {
                    k: float(v) if torch.is_tensor(v) else v for k, v in entry.items()
                }
            error = (fields.cpu().numpy() - true_fields) ** 2
            weights = np.polynomial.legendre.leggauss(common.grid.nlat)[1][::-1]
            metrics["manufactured_field_area_rmse_km"] = float(
                np.sqrt(
                    (error * weights[None, :, None]).sum()
                    / (weights.sum() * common.grid.nlon * common.n_anchors)
                )
            )
            metrics["outside_training_query_support"] = missing_field_metrics(
                true_fields, fields.cpu().numpy(), support, areas
            )
        return metrics

    results["modern_reference"] = score(torch.as_tensor(data["inputs"][:, 3] * 40, device=device))
    for backbone in ("sfno", "functional_tucker"):
        config = copy.deepcopy(common)
        config.model.backbone = backbone
        config.training.latent_gp_weight = 0.001 if backbone == "functional_tucker" else 0
        config.output_dir = str(output / backbone)
        (output / f"{backbone}.json").write_text(json.dumps(config.to_dict(), indent=2))
        summary = train(config)
        predictor = NornPredictor(output / backbone / "best.pt", device)
        results[backbone] = {
            **score(predictor.anchor_fields()),
            "parameters": summary["runtime"]["parameters"],
            "steps": steps,
            "latent_gp_weight": config.training.latent_gp_weight,
            "elapsed_s": summary["elapsed_s"],
        }
    checkpoint = output / "functional_tucker" / "best.pt"
    predictor = NornPredictor(checkpoint, device)
    export_prediction(predictor, [0, 13.4, 60], output / "prediction", ("npz", "netcdf"))
    sample_posterior(
        checkpoint,
        dataset,
        output / "gp_posterior",
        draws=8,
        chains=2,
        burnin=40,
        thin=2,
        beta=0.15,
        seed=123,
        device=device,
    )
    evaluate_posterior(
        checkpoint,
        dataset,
        output / "gp_posterior" / "latent_samples.npz",
        output / "gp_validation.json",
        device=device,
    )

    # Twelve independent manufactured histories on the *same* geometry.
    # Variable slopes/phases are independently drawn, not frames of one history.
    rng = np.random.default_rng(781)
    histories = []
    for _ in range(12):
        slope, phase = rng.uniform(0.005, 0.04), rng.uniform(-50, 50)
        histories.append(
            np.stack(
                [
                    7
                    + 0.8 * np.cos(np.radians(lat_grid)) * np.cos(np.radians(lon_grid + phase))
                    + slope * (60 - age)
                    for age in common.anchor_ages
                ]
            )
        )
    values = np.asarray(histories, dtype=np.float32)
    observed = rng.random(values.shape) < 0.1
    observed[:, 1::2] = False  # Whole unobserved ages as well as spatial holes.
    raw = output / "field_ensemble.npz"
    np.savez_compressed(
        raw,
        thickness_km=np.where(observed, values, np.nan),
        observed_mask=observed,
        latitude=lat[::-1],
        longitude=lon,
        age_ma=common.anchor_ages,
        history_id=[f"manufactured_{i}" for i in range(12)],
        group_id=[f"manufactured_{i}" for i in range(12)],
        split=["train"] * 8 + ["validation"] * 4,
        metadata=np.array(
            json.dumps(
                {
                    "source_semantics": "synthetic",
                    "description": "Independent analytic slopes/phases; shared static synthetic geometry",
                }
            )
        ),
    )
    ensemble = output / "latent_ensemble.npz"
    encode_ensemble(checkpoint, raw, ensemble, steps=encoding_steps, device=device)
    prior = output / "diffusion_prior.pt"
    prior_report = train_latent_prior(
        ensemble,
        prior,
        steps=prior_steps,
        width=32,
        layers=1,
        lengthscale_ma=common.model.latent_lengthscale_ma,
        amplitude=common.model.latent_amplitude,
        device=device,
    )
    sample_posterior(
        checkpoint,
        dataset,
        output / "diffusion_posterior",
        prior=prior,
        draws=2,
        chains=2,
        diffusion_steps=8,
        guidance=0.1,
        device=device,
    )
    evaluate_posterior(
        checkpoint,
        dataset,
        output / "diffusion_posterior" / "latent_samples.npz",
        output / "diffusion_validation.json",
        device=device,
    )
    report = {
        "kind": "manufactured_software_validation",
        "source_semantics": "synthetic",
        "modern_endpoints_allowed": True,
        "split_protocol": data["metadata"]["split_protocol"],
        "records": data["metadata"]["splits"],
        "missing_field_protocol": {
            "definition": "positive-weight training query corners and supervised modern endpoint excluded",
            "modern_conditioning_retained": True,
            "support_artifact": "comparison_support.npz",
            "missing_cells_by_age": (~support).sum((1, 2)).tolist(),
        },
        "models": results,
        "diffusion_prior": prior_report,
        "claim_boundary": "Single manufactured fixture; not real-data accuracy or method superiority",
    }
    write_manifest(output / "comparison.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prior-steps", type=int, default=200)
    parser.add_argument("--encoding-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    run(**vars(parser.parse_args()))
