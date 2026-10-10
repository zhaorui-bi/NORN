"""Public command line interface. Importing help does not require PyTorch."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__


def _config(args):
    from .config import load_config

    config = load_config(args.config)
    tag, constraints = getattr(args, "tag", None), getattr(args, "physics_constraints", None)
    if constraints is not None and tag != "physics":
        raise ValueError("--physics-constraints requires --tag physics")
    if tag is not None:
        config = config.with_tag(tag, constraints)
    if getattr(args, "output", None):
        config.output_dir = str(Path(args.output).resolve())
    if getattr(args, "device", None):
        config.training.device = args.device
    if getattr(args, "steps", None) is not None:
        config.training.steps = args.steps
    return config.validate()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="norn", description="NORN: chronology-aware spherical crustal-thickness inversion"
    )
    parser.add_argument("--version", action="version", version=f"norn-earth {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="Prepare observations, paleo positions, splits and Gauss-grid features"
    )
    prep.add_argument("--config", required=True)
    prep.add_argument("--output", required=True, help="Output dataset NPZ")
    physics = sub.add_parser(
        "prepare-rigid-priors",
        help="Build explicitly assumed rigid-interior trajectory and material-budget priors",
    )
    physics.add_argument("--config", required=True)
    physics.add_argument("--dataset", required=True)
    physics.add_argument("--output", required=True)
    physics.add_argument("--max-regions", type=int, default=48)
    physics.add_argument("--min-edge-distance-km", type=float, default=500)
    train = sub.add_parser(
        "train", help="Fit an instance-specific reconstruction and save standalone checkpoints"
    )
    train.add_argument("--config", required=True)
    train.add_argument(
        "--tag",
        choices=["no_physics", "physics"],
        help="Toggle consistency terms only; both tags share the same prepared geometry/model/data loss",
    )
    train.add_argument(
        "--physics-constraints", help="Explicit priors artifact; requires --tag physics"
    )
    train.add_argument("--output", help="Override the run directory")
    train.add_argument("--device", help="auto, cpu, cuda or cuda:N")
    train.add_argument("--steps", type=int, help="Override total optimizer steps")
    train.add_argument("--resume", help="Resume a compatible checkpoint")
    train.add_argument(
        "--stop-after",
        type=int,
        help="Stop after this many updates and save a resumable checkpoint",
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and display the resolved configuration without fitting",
    )
    infer = sub.add_parser("infer", help="Export thickness maps from a standalone checkpoint")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument(
        "--ages", nargs="+", default=["all"], help="Ages in Ma, or all for every training anchor"
    )
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--formats", nargs="+", choices=["npz", "dat", "netcdf"], default=["npz"])
    animate = sub.add_parser(
        "animate", help="Render moving continents and thickness colours, 0 to 60 Ma"
    )
    animate.add_argument("--checkpoint", required=True)
    animate.add_argument("--output", required=True, help="Output .mp4 or .gif")
    animate.add_argument("--ages", nargs="+", default=["all"])
    animate.add_argument("--device", default="auto")
    animate.add_argument("--fps", type=float, default=6)
    animate.add_argument("--vmin", type=float, default=0)
    animate.add_argument("--vmax", type=float, default=80)
    animate.add_argument("--all-crust", action="store_true", help="Also colour the oceanic crust")
    points = sub.add_parser(
        "predict-points", help="Predict thickness at paleo lon/lat/age CSV queries"
    )
    points.add_argument("--checkpoint", required=True)
    points.add_argument(
        "--csv",
        required=True,
        help="Columns: longitude, latitude, age_ma (coordinates at queried age)",
    )
    points.add_argument("--output", required=True)
    points.add_argument("--device", default="auto")
    evaluate = sub.add_parser(
        "evaluate", help="Score a frozen checkpoint with the same age-marginal observation operator"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    smoke = sub.add_parser(
        "demo",
        help="Generate a reproducible synthetic dataset, full-physics fixture and runnable config",
    )
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .data.preparation import prepare_dataset

            # This output is a dataset file, not a training run directory.
            from .config import load_config

            path = prepare_dataset(load_config(args.config), args.output)
            print(path)
        elif args.command == "prepare-rigid-priors":
            from .config import load_config
            from .physics.preparation import prepare_rigid_priors

            print(
                prepare_rigid_priors(
                    load_config(args.config),
                    args.dataset,
                    args.output,
                    args.max_regions,
                    args.min_edge_distance_km,
                )
            )
        elif args.command == "train":
            config = _config(args)
            if args.dry_run:
                print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
            else:
                from .training.trainer import train

                train(config, resume=args.resume, stop_after=args.stop_after)
        elif args.command == "infer":
            from .inference import NornPredictor, export_prediction

            predictor = NornPredictor(args.checkpoint, args.device)
            ages = (
                predictor.config.anchor_ages
                if args.ages == ["all"]
                else [float(a) for a in args.ages]
            )
            _, meta = export_prediction(predictor, ages, args.output, args.formats)
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        elif args.command == "animate":
            from .inference import NornPredictor
            from .visualization import export_animation

            predictor = NornPredictor(args.checkpoint, args.device)
            ages = (
                predictor.config.anchor_ages
                if args.ages == ["all"]
                else [float(a) for a in args.ages]
            )
            meta = export_animation(
                predictor,
                args.output,
                ages,
                args.fps,
                args.vmin,
                args.vmax,
                continental_only=not args.all_crust,
            )
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        elif args.command == "predict-points":
            import pandas as pd
            from .inference import NornPredictor

            table = pd.read_csv(args.csv)
            needed = ["longitude", "latitude", "age_ma"]
            if any(key not in table for key in needed):
                raise ValueError(f"Point queries require columns {needed}")
            predictor = NornPredictor(args.checkpoint, args.device)
            table["thickness_km"] = predictor.predict_points(
                *(table[key].to_numpy() for key in needed)
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.output, index=False)
            print(args.output)
        elif args.command == "evaluate":
            import torch
            from .data.preparation import load_dataset
            from .inference import NornPredictor
            from .losses.objective import ReconstructionObjective
            from .utils.hashing import sha256_file, write_manifest

            predictor = NornPredictor(args.checkpoint, args.device)
            if sha256_file(args.dataset) != predictor.checkpoint["provenance"]["dataset_sha256"]:
                raise ValueError("Evaluation dataset differs from the frozen training artifact")
            # Evaluation of the observation operator needs no external physics
            # file; this also allows moved standalone checkpoints to be scored.
            predictor.config.physics.constraints = None
            for key in vars(predictor.config.physics):
                if key.endswith("weight"):
                    setattr(predictor.config.physics, key, 0.0)
            objective = ReconstructionObjective(
                load_dataset(args.dataset, predictor.config), predictor.config, predictor.device
            )
            with torch.no_grad():
                metrics = objective.observation_metrics(predictor.anchor_fields(), args.split)
            if metrics is None:
                raise ValueError(f"The frozen dataset has no {args.split} records")
            report = {
                "split": args.split,
                "metrics": {
                    k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()
                },
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "dataset_sha256": sha256_file(args.dataset),
                "source_semantics": predictor.checkpoint["provenance"]["dataset_metadata"][
                    "source_semantics"
                ],
                "modern_endpoints_allowed": True,
            }
            write_manifest(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "demo":
            from .data.demo import create_demo

            print(create_demo(args.output, args.device))
        return 0
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        ImportError,
        RuntimeError,
        FloatingPointError,
    ) as exc:
        print(f"norn: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
