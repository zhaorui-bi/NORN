"""Portable, validated JSON configurations. Relative paths are config-relative."""

import copy
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Optional


@dataclass
class GridConfig:
    nlat: int = 180
    nlon: int = 360


@dataclass
class ModelConfig:
    backbone: str = "sfno"
    head_mode: str = "absolute"
    time_frequency_bands: int = 2
    ffn_multiplier: int = 2
    width: int = 32
    blocks: int = 6
    lmax: int = 32
    mmax: int = 32
    in_channels: int = 8
    time_features: int = 32
    epsilon_km: float = 0.5
    initial_thickness_km: float = 30.0


@dataclass
class DataConfig:
    prepared: Optional[str] = None
    observations: Optional[str] = None
    modern: Optional[str] = None
    rotations: Optional[str] = None
    static_polygons: Optional[str] = None
    continental_polygons: Optional[str] = None
    topologies: Optional[str] = None
    geometry_mode: str = "static"
    geometry_source: str = "gpml"
    plate_gmt_dir: Optional[str] = None
    deformation_gmt_dir: Optional[str] = None
    coordinate_mode: str = "paleo"
    source_semantics: str = "unverified_proxy"
    holdout_csv: Optional[str] = None
    holdout_column: str = "chal_union"
    validation_fraction: float = 0.1
    record_sigma_km: float = 4.0
    continental_sigma_km: float = 9.0
    thin_crust_sigma_km: float = 26.0
    modern_sigma_km: float = 3.0
    include_flagged_records: bool = False
    position_failure: str = "drop"


@dataclass
class PhysicsConfig:
    constraints: Optional[str] = None
    trajectory_weight: float = 0.0
    budget_weight: float = 0.0
    birth_weight: float = 0.0
    source_prior_weight: float = 0.0
    source_smooth_weight: float = 0.0


@dataclass
class TrainingConfig:
    objective: str = "data_only"
    selection_metric: str = "validation_mae"
    observation_huber_weight: float = 0.0
    huber_beta_km: float = 5.0
    steps: int = 500
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    warmup_steps: int = 25
    gradient_clip: float = 1.0
    anchor_batch_size: int = 1
    modern_weight: float = 0.5
    observation_weight: float = 1.0
    temporal_smooth_weight: float = 0.0
    checkpoint_every: int = 50
    log_every: int = 10
    validation_every: int = 10
    seed: int = 42
    device: str = "auto"
    num_threads: int = 4


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    max_age_ma: float = 60.0
    anchor_step_myr: float = 1.0
    output_dir: str = "outputs/run"
    schema_version: int = 1

    @property
    def anchor_ages(self):
        import numpy as np

        return np.arange(self.n_anchors, dtype=np.float32) * self.anchor_step_myr

    @property
    def n_anchors(self):
        return round(self.max_age_ma / self.anchor_step_myr) + 1

    def to_dict(self):
        return asdict(self)

    @property
    def experiment_tag(self):
        return "no_physics" if self.training.objective == "data_only" else "physics"

    def with_tag(self, tag, physics_constraints=None):
        """Change consistency terms only; preserve model, data loss and schedule.

        Tag runs get separate directories unless the CLI explicitly overrides
        --output. Enabling physics requires a declared artifact; no silent
        zero-weight 'physics' experiment is allowed.
        """
        config = copy.deepcopy(self)
        if tag == "no_physics":
            if physics_constraints is not None:
                raise ValueError("--physics-constraints requires --tag physics")
            config.training.objective = "data_only"
            config.training.temporal_smooth_weight = 0.0
            config.physics = PhysicsConfig()
        elif tag == "physics":
            config.training.objective = "reconstruction"
            if physics_constraints is not None:
                config.physics.constraints = str(Path(physics_constraints).expanduser().resolve())
            if not config.physics.constraints:
                raise ValueError(
                    "--tag physics requires --physics-constraints or physics.constraints"
                )
            if not any(
                getattr(config.physics, item.name) > 0
                for item in fields(config.physics)
                if item.name.endswith("weight")
            ):
                config.physics.trajectory_weight = 0.05
                config.physics.budget_weight = 0.05
        else:
            raise ValueError("tag must be no_physics or physics")
        root = Path(config.output_dir)
        if root.name in ("no_physics", "physics"):
            root = root.parent
        config.output_dir = str(root / tag)
        return config.validate()

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema_version")
        g, m, d, t, p = self.grid, self.model, self.data, self.training, self.physics
        if g.nlat < 4 or g.nlon < 8:
            raise ValueError("Grid must have at least 4 latitude and 8 longitude nodes")
        if not (1 <= m.mmax <= m.lmax <= g.nlat and m.mmax <= g.nlon // 2 + 1):
            raise ValueError("Require 1 <= mmax <= lmax <= nlat and mmax <= nlon/2+1")
        if m.in_channels != 8 or m.width < 2 or m.blocks < 1 or m.time_features < 2:
            raise ValueError(
                "Model requires 8 input channels, width>=2, blocks>=1, time_features>=2"
            )
        if m.backbone not in ("sfno", "gated_sfno"):
            raise ValueError("model.backbone must be sfno or gated_sfno")
        if m.head_mode not in ("absolute", "input_residual"):
            raise ValueError("model.head_mode must be absolute or input_residual")
        if m.time_frequency_bands < 1 or m.ffn_multiplier < 1:
            raise ValueError("time_frequency_bands and ffn_multiplier must be >= 1")
        if m.epsilon_km <= 0 or m.initial_thickness_km <= m.epsilon_km:
            raise ValueError("Require initial_thickness_km > epsilon_km > 0")
        if not 0 < self.max_age_ma <= 60 or self.anchor_step_myr <= 0:
            raise ValueError("Require 0 < max_age_ma <= 60 and a positive anchor step")
        ratio = self.max_age_ma / self.anchor_step_myr
        if not math.isclose(ratio, round(ratio), abs_tol=1e-7) or self.n_anchors < 2:
            raise ValueError("max_age_ma must be a positive integer multiple of anchor_step_myr")
        if d.geometry_mode not in ("static", "dynamic"):
            raise ValueError("geometry_mode must be static or dynamic")
        if d.geometry_source not in ("gpml", "gmt"):
            raise ValueError("geometry_source must be gpml or gmt")
        if d.geometry_source == "gmt" and (
            d.geometry_mode != "dynamic" or not d.plate_gmt_dir or not d.deformation_gmt_dir
        ):
            raise ValueError("GMT geometry requires dynamic mode and both GMT snapshot directories")
        if d.geometry_source != "gmt" and (d.plate_gmt_dir or d.deformation_gmt_dir):
            raise ValueError(
                "Declared GMT sources require geometry_source=gmt; refusing to ignore them"
            )
        if d.geometry_mode == "dynamic":
            if d.coordinate_mode != "paleo":
                raise ValueError("Dynamic geometry requires paleo observation coordinates")
            if not all((d.rotations, d.static_polygons, d.continental_polygons, d.topologies)):
                raise ValueError(
                    "Dynamic geometry requires rotations, static_polygons, "
                    "continental_polygons and topologies from a consistent reconstruction model"
                )
        if d.coordinate_mode not in ("paleo", "present"):
            raise ValueError("coordinate_mode must be paleo or present")
        if d.position_failure not in ("drop", "error"):
            raise ValueError(
                "position_failure must be drop or error; silent present-coordinate fallback is forbidden"
            )
        if d.source_semantics not in ("unverified_proxy", "verified_thickness", "synthetic"):
            raise ValueError("Declare source_semantics explicitly")
        if not 0 <= d.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        for name in (
            "record_sigma_km",
            "continental_sigma_km",
            "thin_crust_sigma_km",
            "modern_sigma_km",
        ):
            if not getattr(d, name) > 0:
                raise ValueError(f"data.{name} must be positive")
        for name in (
            "steps",
            "anchor_batch_size",
            "checkpoint_every",
            "log_every",
            "validation_every",
            "num_threads",
        ):
            if getattr(t, name) < 1:
                raise ValueError(f"training.{name} must be >= 1")
        if t.learning_rate <= 0 or t.gradient_clip <= 0 or t.warmup_steps < 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if t.objective not in ("reconstruction", "data_only"):
            raise ValueError("training.objective must be reconstruction or data_only")
        if t.selection_metric not in ("joint_nll", "validation_nll", "validation_mae"):
            raise ValueError("Unknown training.selection_metric")
        if not math.isfinite(t.huber_beta_km) or t.huber_beta_km <= 0:
            raise ValueError("huber_beta_km must be finite and positive")
        if t.objective == "data_only" and (
            p.constraints is not None
            or any(getattr(p, f.name) != 0 for f in fields(p) if f.name.endswith("weight"))
            or t.temporal_smooth_weight != 0
        ):
            raise ValueError(
                "data_only forbids physics constraints artifacts/weights and temporal smoothing"
            )
        for obj, names in (
            (
                t,
                (
                    "modern_weight",
                    "observation_weight",
                    "temporal_smooth_weight",
                    "observation_huber_weight",
                ),
            ),
            (
                p,
                (
                    "trajectory_weight",
                    "budget_weight",
                    "birth_weight",
                    "source_prior_weight",
                    "source_smooth_weight",
                ),
            ),
        ):
            for name in names:
                if not math.isfinite(getattr(obj, name)) or getattr(obj, name) < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
        if (
            any(getattr(p, f.name) > 0 for f in fields(p) if f.name.endswith("weight"))
            and not p.constraints
        ):
            raise ValueError("Physics weights require an explicit constraints artifact")
        return self


_SECTIONS = {
    "grid": GridConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "physics": PhysicsConfig,
    "training": TrainingConfig,
}
_PATHS = {
    "data": (
        "prepared",
        "observations",
        "modern",
        "rotations",
        "static_polygons",
        "continental_polygons",
        "topologies",
        "plate_gmt_dir",
        "deformation_gmt_dir",
        "holdout_csv",
    ),
    "physics": ("constraints",),
}


def config_from_dict(payload, base_dir=None):
    body = dict(payload)
    unknown = set(body) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name, cls in _SECTIONS.items():
        values = dict(body.get(name, {}))
        extra = set(values) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown {name} keys: {sorted(extra)}")
        if base_dir is not None:
            for key in _PATHS.get(name, ()):
                if values.get(key):
                    path = Path(values[key]).expanduser()
                    values[key] = str((Path(base_dir) / path).resolve())
        body[name] = cls(**values)
    if base_dir is not None:
        body["output_dir"] = str((Path(base_dir) / body.get("output_dir", "outputs/run")).resolve())
    return Config(**body).validate()


def load_config(path):
    path = Path(path).resolve()
    return config_from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)
