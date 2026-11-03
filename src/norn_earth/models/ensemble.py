"""A shared latent coordinate system and history-level split for learned priors."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..geometry.regrid import gauss_grid
from ..training.trainer import load_checkpoint, resolve_device
from ..utils.hashing import sha256_file, write_manifest


def decoder_fingerprint(model, inputs, config):
    """Bind a prior to the exact frozen factors, geometry, transform and ages.

    Separately trained Tucker models have an arbitrary latent gauge and must
    never have their cores pooled, even if their rank/shape are identical.
    """
    h = hashlib.sha256()
    semantics = {
        "model": config.to_dict()["model"],
        "grid": config.to_dict()["grid"],
        "ages_ma": config.anchor_ages.tolist(),
    }
    h.update(json.dumps(semantics, sort_keys=True).encode())
    tensors = {k: v for k, v in model.state_dict().items() if k != "cores"}
    tensors["__age_bound_inputs__"] = inputs
    for key, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def validate_history_split(ids, groups, split):
    ids, groups, split = (np.char.strip(np.asarray(x).astype(str)) for x in (ids, groups, split))
    if ids.ndim != 1 or groups.shape != ids.shape or split.shape != ids.shape:
        raise ValueError("history_id, group_id and split must be equally sized 1-D arrays")
    if any(not x.strip() for x in ids) or any(not x.strip() for x in groups):
        raise ValueError("History and independent parent group IDs must be nonempty")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("history_id must be unique; use group_id for related windows")
    if not np.isin(split, ["train", "validation", "test"]).all():
        raise ValueError("History split must be train, validation or test")
    sets = {key: set(groups[split == key]) for key in ("train", "validation", "test")}
    if any(
        sets[a] & sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Related windows from a parent history cross ensemble splits")
    if len(sets["train"]) < 2 or len(sets["validation"]) < 1:
        raise ValueError("Need >=2 independent training histories and >=1 validation history")
    return ids, groups, split


def load_latent_ensemble(path):
    with np.load(path, allow_pickle=False) as bundle:
        needed = {"cores", "age_ma", "history_id", "group_id", "split", "metadata"}
        if not needed.issubset(bundle.files):
            raise ValueError(f"Latent ensemble requires {sorted(needed)}")
        data = {key: bundle[key] for key in needed}
    return _validate_latent_ensemble(data)


def _validate_latent_ensemble(data):
    meta = json.loads(str(data["metadata"]))
    if meta.get("source_semantics") not in ("synthetic", "verified_simulation"):
        raise ValueError(
            "A learned prior requires declared synthetic or verified_simulation histories"
        )
    if meta.get("schema_version") != 1 or not meta.get("decoder_sha256"):
        raise ValueError("Latent ensemble needs schema_version=1 and a shared decoder fingerprint")
    cores, ages = data["cores"], data["age_ma"]
    if (
        cores.ndim != 4
        or ages.ndim != 1
        or cores.shape[1] != len(ages)
        or len(ages) < 2
        or not np.isfinite(cores).all()
        or not np.isfinite(ages).all()
        or np.any(np.diff(ages) <= 0)
    ):
        raise ValueError(
            "Expected finite cores[history,age,spatial_rank,context_rank] and sorted ages"
        )
    ids, groups, split = validate_history_split(data["history_id"], data["group_id"], data["split"])
    if len(ids) != len(cores):
        raise ValueError("History IDs differ from the number of core sequences")
    # Prevent literal duplicate trajectories being relabelled as independent.
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in cores]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate core histories cannot form independent prior examples")
    return {**data, "history_id": ids, "group_id": groups, "split": split, "metadata": meta}


def encode_ensemble(
    checkpoint, fields_path, output, steps=200, learning_rate=0.03, seed=42, device="auto"
):
    """Fit only cores for supplied histories; decoder and age-bound geometry stay frozen.

    Fields use the checkpoint's north-to-south Gauss grid. `observed_mask`
    allows sparse fields; unobserved values may be NaN and never enter loss.
    Each history must have support at some ages; GP regularization fills gaps.
    """
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Encoding steps and learning rate must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = resolve_device(device)
    _, config, model, inputs = load_checkpoint(checkpoint, device)
    if config.model.backbone != "functional_tucker":
        raise ValueError("Ensemble encoding requires a functional_tucker checkpoint")
    torch.set_num_threads(config.training.num_threads)
    torch.manual_seed(seed)
    with np.load(fields_path, allow_pickle=False) as bundle:
        needed = {
            "thickness_km",
            "observed_mask",
            "age_ma",
            "latitude",
            "longitude",
            "history_id",
            "group_id",
            "split",
            "metadata",
        }
        if not needed.issubset(bundle.files):
            raise ValueError(f"Field ensemble requires {sorted(needed)}")
        data = {k: bundle[k] for k in needed}
    ids, groups, split = validate_history_split(data["history_id"], data["group_id"], data["split"])
    meta = json.loads(str(data["metadata"]))
    if meta.get("source_semantics") not in ("synthetic", "verified_simulation"):
        raise ValueError("Declare ensemble source_semantics as synthetic or verified_simulation")
    lat, lon, _ = gauss_grid(config.grid.nlat, config.grid.nlon)
    if (
        not np.array_equal(data["age_ma"].astype(np.float32), config.anchor_ages)
        or data["latitude"].shape != lat.shape
        or data["longitude"].shape != lon.shape
        or not np.allclose(data["latitude"], lat[::-1])
        or not np.allclose(data["longitude"], lon)
    ):
        raise ValueError(
            "Ensemble coordinates differ from checkpoint ages/north-to-south Gauss grid"
        )
    values, mask = data["thickness_km"], data["observed_mask"]
    expected = (len(ids), config.n_anchors, config.grid.nlat, config.grid.nlon)
    if values.shape != expected or mask.shape != expected or not np.isin(mask, [0, 1]).all():
        raise ValueError(
            "Ensemble values/mask must match history x checkpoint-age x latitude x longitude"
        )
    mask = mask.astype(bool)
    if (
        not mask.reshape(len(ids), -1).any(1).all()
        or not np.isfinite(values[mask]).all()
        or np.any(values[mask] <= 0)
    ):
        raise ValueError("Each history needs finite positive observed thickness")
    model.eval().requires_grad_(False)
    with torch.no_grad():
        factors = tuple(x.detach() for x in model.encode_factors(inputs))
    encoded, errors = [], []
    for index, history in enumerate(ids):
        y = torch.as_tensor(
            np.where(mask[index], values[index], 0), device=device, dtype=torch.float32
        )
        observed = torch.as_tensor(mask[index], device=device)
        core = torch.nn.Parameter(model.cores.detach().clone())
        optimizer = torch.optim.Adam([core], lr=learning_rate)
        from .temporal import whiten_cores

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.decode_factors(factors, core)[:, 0]
            mse = (prediction[observed] - y[observed]).square().mean()
            loss = mse + 1e-5 * whiten_cores(core, model.prior_cholesky).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ensemble encoding loss")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = model.decode_factors(factors, core)[:, 0]
            errors.append(float((pred[observed] - y[observed]).square().mean().sqrt()))
        encoded.append(core.detach().cpu().numpy())
        if index == 0 or index == len(ids) - 1 or (index + 1) % 10 == 0:
            print(f"encoded {index + 1}/{len(ids)} histories", flush=True)
    metadata = {
        "schema_version": 1,
        "source_semantics": meta["source_semantics"],
        "decoder_sha256": decoder_fingerprint(model, inputs, config),
        "fields_sha256": sha256_file(fields_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "shared_geometry": True,
        "encoding_steps": steps,
        "encoding_seed": seed,
        "encoding_learning_rate": learning_rate,
        "encoding_gp_weight": 0.00001,
        "observed_rmse_km": dict(zip(ids.tolist(), errors)),
        "independence_unit": "declared parent physical history (group_id)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cores": np.stack(encoded),
        "age_ma": config.anchor_ages,
        "history_id": ids,
        "group_id": groups,
        "split": split,
        "metadata": np.array(json.dumps(metadata)),
    }
    _validate_latent_ensemble(payload)
    # Validate before publishing; a file handle prevents silently appending .npz.
    with output.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    write_manifest(output.with_suffix(".json"), metadata)
    return metadata
