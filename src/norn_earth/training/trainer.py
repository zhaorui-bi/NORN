"""Memory-bounded two-pass training and resumable, standalone checkpoints."""

import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .. import __version__
from ..config import config_from_dict
from ..data.preparation import load_dataset, prepare_dataset
from ..losses.objective import ReconstructionObjective
from ..models.operator import NornSFNO
from ..utils.hashing import sha256_file, write_manifest

CHECKPOINT_SCHEMA = 1


def resolve_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu or install a CUDA PyTorch build"
        )
    return device


def anchor_fields(model, inputs, ages, batch_size):
    fields = []
    for start in range(0, len(ages), batch_size):
        age = ages[start : start + batch_size].reshape(-1, 1)
        fields.append(model(inputs.expand(len(age), -1, -1, -1), age)[:, 0])
    return torch.cat(fields, dim=0)


def recompute_backward(model, inputs, ages, field_gradient, batch_size):
    # Backward immediately per chunk. Summing 61 graph-carrying surrogate
    # tensors and calling backward once would retain every activation graph.
    for start in range(0, len(ages), batch_size):
        age = ages[start : start + batch_size].reshape(-1, 1)
        h = model(inputs.expand(len(age), -1, -1, -1), age)[:, 0]
        (h * field_gradient[start : start + len(age)]).sum().backward()


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Unsupported or legacy checkpoint; historical A40 weights require retraining"
        )
    config = config_from_dict(checkpoint["config"])
    model = NornSFNO(config.grid, config.model, config.max_age_ma).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    inputs = checkpoint["inputs"].to(device)
    if (
        inputs.shape != (1, 8, config.grid.nlat, config.grid.nlon)
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError("Invalid checkpoint input features")
    return checkpoint, config, model, inputs


def _save(path, model, objective, optimizer, config, inputs, step, best, provenance, metrics):
    body = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "software_version": __version__,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "physics_state": objective.physics.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "inputs": inputs.detach().cpu(),
        "completed_steps": int(step),
        "best_validation_score": float(best),
        "provenance": provenance,
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(body, temporary)
    os.replace(temporary, path)


def _numbers(values):
    return {
        key: float(value.detach()) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def train(config, resume=None, stop_after=None):
    """Optimize one observation collection. No external protocol files needed."""
    config.validate()
    t = config.training
    torch.set_num_threads(t.num_threads)
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    device = resolve_device(t.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(config.data.prepared) if config.data.prepared else output / "dataset.npz"
    if not dataset.is_file():
        if config.data.prepared:
            raise FileNotFoundError(dataset)
        print("Preparing observations, reconstructed positions and spherical inputs...", flush=True)
        prepare_dataset(config, dataset)
    data = load_dataset(dataset, config)
    inputs = torch.as_tensor(data["inputs"], device=device).unsqueeze(0)
    model = NornSFNO(config.grid, config.model, config.max_age_ma).to(device)
    objective = ReconstructionObjective(data, config, device)
    parameters = list(model.parameters()) + list(objective.physics.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=t.learning_rate, weight_decay=t.weight_decay)
    ages = torch.as_tensor(config.anchor_ages, device=device)
    provenance = {
        "dataset_sha256": sha256_file(dataset),
        "dataset_metadata": data["metadata"],
        "physics": objective.physics.metadata,
        "physics_sha256": sha256_file(config.physics.constraints)
        if config.physics.constraints
        else None,
    }
    start = 0
    best = float("inf")
    if resume:
        checkpoint, previous, _, _ = load_checkpoint(resume, device)
        if checkpoint["provenance"]["dataset_sha256"] != provenance["dataset_sha256"]:
            raise ValueError("Resume dataset differs from the frozen checkpoint dataset")
        for key in ("grid", "model", "max_age_ma", "anchor_step_myr", "physics"):
            if previous.to_dict()[key] != config.to_dict()[key]:
                raise ValueError(f"Cannot resume with a changed {key} configuration")
        adjustable = {
            "steps",
            "device",
            "num_threads",
            "log_every",
            "checkpoint_every",
            "validation_every",
        }
        for key, value in vars(previous.training).items():
            if key not in adjustable and value != getattr(config.training, key):
                raise ValueError(f"Cannot resume with changed training.{key}; start a new run")
        if checkpoint["provenance"]["physics_sha256"] != provenance["physics_sha256"]:
            raise ValueError("Physics artifact changed since the checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        objective.physics.load_state_dict(checkpoint["physics_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng_states"]])
        start = checkpoint["completed_steps"]
        best = checkpoint["best_validation_score"]
    elif (output / "last.pt").exists():
        raise FileExistsError(
            f"{output}/last.pt exists; pass --resume or choose a new output directory"
        )
    if start >= t.steps:
        raise ValueError(
            "Checkpoint already reached training.steps; increase the requested steps to continue"
        )
    runtime = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "parameters": sum(p.numel() for p in parameters),
        "software_version": __version__,
    }
    write_manifest(
        output / "run_manifest.json",
        {"config": config.to_dict(), "runtime": runtime, "provenance": provenance},
    )
    print(
        json.dumps(
            {
                "runtime": runtime,
                "records": data["metadata"]["splits"],
                "physics": objective.physics.metadata.get("active_factors", {}),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    began = time.monotonic()
    initial_loss = None
    last_metrics = {}
    end = min(t.steps, start + stop_after) if stop_after is not None else t.steps
    if end <= start:
        raise ValueError("stop_after must be positive")
    logpath = output / "training_log.csv"
    log_handle = None
    try:
        for step in range(start, end):
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                cached = anchor_fields(model, inputs, ages, t.anchor_batch_size)
            leaf = cached.detach().requires_grad_(True)
            loss, parts = objective(leaf)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at step {step}")
            if initial_loss is None:
                initial_loss = float(loss.detach())
            # Source parameters receive their direct joint-objective gradients
            # here; model parameters receive the chain-rule gradients below.
            loss.backward()
            gradient = leaf.grad.detach()
            last_metrics = {"step": step, "loss": float(loss.detach()), **_numbers(parts)}
            validation = step % t.validation_every == 0 or step == start
            if validation:
                with torch.no_grad():
                    metrics = objective.observation_metrics(cached, "validation")
                if metrics:
                    last_metrics.update(
                        {f"validation_{k}": v for k, v in _numbers(metrics).items()}
                    )
                    score = float(metrics["nll"]) + t.modern_weight * float(
                        parts["modern_nll"].detach()
                    )
                else:
                    score = float(loss.detach())
                if score < best:
                    best = score
                    _save(
                        output / "best.pt",
                        model,
                        objective,
                        optimizer,
                        config,
                        inputs,
                        step,
                        best,
                        provenance,
                        last_metrics,
                    )
            # Parameters stay fixed between the cache and every recomputation.
            recompute_backward(model, inputs, ages, gradient, t.anchor_batch_size)
            gradnorm = torch.nn.utils.clip_grad_norm_(
                parameters, t.gradient_clip, error_if_nonfinite=True
            )
            warm = min(1.0, (step + 1) / max(1, t.warmup_steps))
            phase = max(0, (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps))
            lr = t.learning_rate * warm * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * phase)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            last_metrics.update(
                {
                    "learning_rate": lr,
                    "gradient_norm": float(gradnorm),
                    "time_s": time.monotonic() - tick,
                }
            )
            if step % t.log_every == 0 or step == end - 1:
                print(
                    f"step {step:5d}/{t.steps}: loss={last_metrics['loss']:.6f} "
                    f"obs={last_metrics['observation_nll']:.5f} modern={last_metrics['modern_nll']:.5f} "
                    f"physics={last_metrics['physics_loss']:.6f} time={last_metrics['time_s']:.2f}s",
                    flush=True,
                )
            if log_handle is None:
                log_handle = logpath.open("a" if resume else "w", newline="", encoding="utf-8")
                keys = [
                    "step",
                    "loss",
                    *parts,
                    "validation_nll",
                    "validation_age_marginal_mean_mae_km",
                    "validation_n_records",
                    "learning_rate",
                    "gradient_norm",
                    "time_s",
                ]
                writer = csv.DictWriter(log_handle, fieldnames=keys, extrasaction="ignore")
                if not resume:
                    writer.writeheader()
            writer.writerow(last_metrics)
            log_handle.flush()
            if (step + 1) % t.checkpoint_every == 0:
                _save(
                    output / "last.pt",
                    model,
                    objective,
                    optimizer,
                    config,
                    inputs,
                    step + 1,
                    best,
                    provenance,
                    {},
                )
        # Score the saved state itself, not the previous optimizer state.
        model.eval()
        with torch.no_grad():
            final_fields = anchor_fields(model, inputs, ages, t.anchor_batch_size)
            final_loss, final_parts = objective(final_fields)
            final_metrics = {"step": end, "loss": float(final_loss), **_numbers(final_parts)}
            val = objective.observation_metrics(final_fields, "validation")
            if val:
                final_metrics.update({f"validation_{k}": v for k, v in _numbers(val).items()})
                final_score = float(val["nll"]) + t.modern_weight * float(final_parts["modern_nll"])
            else:
                final_score = float(final_loss)
        if final_score < best:
            best = final_score
            _save(
                output / "best.pt",
                model,
                objective,
                optimizer,
                config,
                inputs,
                end,
                best,
                provenance,
                final_metrics,
            )
        _save(
            output / "last.pt",
            model,
            objective,
            optimizer,
            config,
            inputs,
            end,
            best,
            provenance,
            final_metrics,
        )
        report = {
            "completed_steps": end,
            "requested_steps": t.steps,
            "completed": end == t.steps,
            "initial_loss_this_invocation": initial_loss,
            "final": final_metrics,
            "best_validation_score": best,
            "elapsed_s": time.monotonic() - began,
            "runtime": runtime,
            "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0,
            "checkpoint_sha256": sha256_file(output / "last.pt"),
            "source_coefficients_km_myr": objective.physics.sources().detach().cpu().tolist()
            if objective.physics.sources is not None
            else [],
        }
        write_manifest(output / "training_summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
    finally:
        if log_handle is not None:
            log_handle.close()
