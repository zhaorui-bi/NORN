"""Moving-plate thickness movies from frozen, age-matched predictions.

No fixed modern coastline is drawn. Colour limits are fixed across the entire
movie; geometry comes from the same checkpoint inputs used by the network.
"""

from pathlib import Path
import subprocess

import numpy as np

from .utils.hashing import sha256_file, write_manifest


def plate_edges(ids):
    """Raster plate edges; longitude wraps, latitude does not."""
    ids = np.asarray(ids)
    edge = np.zeros(ids.shape, dtype=bool)
    neighbour = np.roll(ids, 1, axis=1)
    edge |= (ids >= 0) & (neighbour >= 0) & (ids != neighbour)
    edge[1:] |= (ids[1:] >= 0) & (ids[:-1] >= 0) & (ids[1:] != ids[:-1])
    return edge


def export_animation(
    predictor,
    output,
    ages=None,
    fps=6,
    vmin=0.0,
    vmax=80.0,
    continental_only=True,
    cmap="turbo",
    dpi=100,
):
    """Write MP4 (system ffmpeg) or GIF (Pillow), plus preview and provenance.

    Default playback is 0 -> 60 Ma, backwards in geological time, NOT the
    training optimizer's iterations or diffusion denoising steps. Fractional
    frames interpolate frozen anchor fields; they are not new observations.
    """
    if predictor.config.data.geometry_mode != "dynamic":
        raise ValueError("Moving-plate movies require a dynamic-geometry checkpoint; retrain first")
    if not np.isfinite([fps, vmin, vmax, dpi]).all() or fps <= 0 or dpi <= 0 or vmax <= vmin:
        raise ValueError("Require finite fps>0, dpi>0 and vmax>vmin")
    ages = np.asarray(predictor.config.anchor_ages if ages is None else ages, dtype=float)
    if ages.ndim != 1 or len(ages) < 2 or np.any(np.diff(ages) <= 0):
        raise ValueError("Movie ages must be strictly increasing, with at least two frames")
    output = Path(output)
    if output.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("Movie output must end in .mp4 or .gif")
    if output.exists():
        raise FileExistsError(output)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, PillowWriter
    except ImportError as exc:
        raise ImportError("Animation requires the viz extra (matplotlib)") from exc
    if output.suffix.lower() == ".mp4" and not FFMpegWriter.isAvailable():
        raise RuntimeError("MP4 requires ffmpeg on PATH; use .gif otherwise")
    codec = None
    if output.suffix.lower() == ".mp4":
        encoders = subprocess.run(
            [matplotlib.rcParams["animation.ffmpeg_path"], "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        available = {
            line.split()[1]
            for line in encoders.splitlines()
            if len(line.split()) >= 2 and line.split()[0].startswith("V")
        }
        # Some distribution builds list an encoder whose runtime library is
        # missing/broken. Probe a real frame instead of trusting the list.
        for candidate in ("libx264", "libopenh264", "mpeg4"):
            if candidate not in available:
                continue
            trial = subprocess.run(
                [
                    matplotlib.rcParams["animation.ffmpeg_path"],
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=size=320x240:rate={fps}",
                    "-frames:v",
                    "1",
                    "-c:v",
                    candidate,
                    "-b:v",
                    "2500k",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "2",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                timeout=15,
            )
            if trial.returncode == 0:
                codec = candidate
                break
        if codec is None:
            raise RuntimeError("ffmpeg has no working MP4 encoder; use .gif")
    result = predictor.predict_grid(ages)
    lon = (result["longitude"] + 180) % 360 - 180
    order = np.argsort(lon)
    lon, lat = lon[order], result["latitude"]
    fields = result["thickness_km"][:, :, order]
    continents = result["continental_fraction"][:, :, order]
    ids = result["plate_id"][:, :, order]

    def draw(ax, index):
        ax.clear()
        ax.set_facecolor("#102238")
        field = (
            np.ma.masked_where(continents[index] < 0.5, fields[index])
            if continental_only
            else fields[index]
        )
        image = ax.imshow(
            field,
            origin="lower",
            extent=(-180, 180, -90, 90),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            aspect="auto",
        )
        edges = plate_edges(ids[index])
        if edges.any() and not edges.all():
            ax.contour(lon, lat, edges, levels=[0.5], colors="#9eafbd", linewidths=0.4, alpha=0.6)
        if continents[index].min() < 0.5 < continents[index].max():
            ax.contour(lon, lat, continents[index], levels=[0.5], colors="white", linewidths=0.65)
        ax.set(
            xlim=(-180, 180),
            ylim=(-90, 90),
            xlabel="Longitude",
            ylabel="Latitude",
            title=f"Crustal-thickness reconstruction | {ages[index]:g} Ma",
        )
        return image

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial" + output.suffix)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    image = draw(ax, 0)
    fig.colorbar(image, ax=ax, label="Crustal thickness (km)", extend="max")
    fig.suptitle("Present → past | White: reconstructed continental outlines | Grey: plate edges")
    writer = (
        FFMpegWriter(
            fps=fps,
            codec=codec,
            bitrate=2500,
            extra_args=["-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart"],
        )
        if output.suffix.lower() == ".mp4"
        else PillowWriter(fps=fps)
    )
    try:
        with writer.saving(fig, str(temporary), dpi=dpi):
            for index in range(len(ages)):
                draw(ax, index)
                writer.grab_frame()
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    preview, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    try:
        for ax, index in zip(axes, [0, len(ages) // 2, len(ages) - 1]):
            image = draw(ax, index)
        preview.colorbar(image, ax=axes, label="Crustal thickness (km)", extend="max")
        preview.savefig(output.with_suffix(".png"), dpi=dpi)
    finally:
        plt.close(preview)
    metadata = {
        "video": output.name,
        "codec": codec if codec is not None else "gif",
        "frames": len(ages),
        "fps": float(fps),
        "ages_ma": ages.tolist(),
        "direction": "0 towards 60 Ma (present to past)",
        "colour_limits_km": [float(vmin), float(vmax)],
        "colormap": cmap,
        "continental_only_display": continental_only,
        "geometry_mode": "dynamic",
        "geometry_source": predictor.config.data.geometry_source,
        "experiment_tag": predictor.config.experiment_tag,
        "coastlines": "checkpoint reconstructed continental mask",
        "plate_edges": "nearest-anchor categorical plate IDs, raster resolution",
        "fractional_frames": "anchor interpolation, not independently resolved topologies",
        "thickness": "model point estimates; not rigidly recoloured modern thickness",
        "checkpoint_sha256": sha256_file(predictor.checkpoint_path),
        "video_sha256": sha256_file(output),
        "geometry": predictor.checkpoint["provenance"]["dataset_metadata"]["geometry"],
    }
    write_manifest(output.with_suffix(".json"), metadata)
    return metadata
