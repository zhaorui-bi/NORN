"""Parse already-reconstructed GMT polygons without losing parts or holes.

Snapshot metadata identifies ages, reference frames and features; it is never a
thickness label, velocity, strain or ridge/trench classification. Coordinates
must not be rotated a second time when rasterizing an exported snapshot.
"""

import csv
from pathlib import Path
import re

import numpy as np


def parse_gmt(path):
    """Preserve @D attributes, @P exterior parts and @H interior rings.

    ``lon``/``lat`` remain aliases for the first exterior ring. An unspecified
    ring role is retained, not silently interpreted as a hole. Malformed data
    raise with a file/line diagnostic instead of dropping vertices/features.
    """
    features, current, fields, geometry_type = [], None, [], None
    with open(path, encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            text = line.strip()
            try:
                if text.startswith("# @VGMT"):
                    match = re.search(r"@G(\w+)", text)
                    geometry_type = match.group(1) if match else None
                elif text.startswith("# @N"):
                    fields = text[4:].split("|")
                    if len(set(fields)) != len(fields):
                        raise ValueError("Duplicate GMT attribute names")
                elif text.startswith("# @D"):
                    values = next(csv.reader([text[4:]], delimiter="|", strict=True))
                    if not fields or len(values) != len(fields):
                        raise ValueError("GMT attribute count differs from @N header")
                    current = {"meta": dict(zip(fields, values)), "rings": [], "ring_roles": []}
                    features.append(current)
                elif text.startswith(">"):
                    if current is not None:
                        current["rings"].append([])
                        current["ring_roles"].append("unspecified")
                elif text in ("# @P", "# @H"):
                    if current is None:
                        raise ValueError("Ring marker before @D feature")
                    if not current["rings"] or current["rings"][-1]:
                        current["rings"].append([])
                        current["ring_roles"].append("unspecified")
                    current["ring_roles"][-1] = "exterior" if text == "# @P" else "hole"
                elif text and not text.startswith("#"):
                    if current is None:
                        raise ValueError("Coordinates before @D feature")
                    parts = text.split()
                    if len(parts) < 2:
                        raise ValueError("Expected longitude and latitude")
                    point = (float(parts[0]), float(parts[1]))
                    if not np.isfinite(point).all() or abs(point[1]) > 90:
                        raise ValueError("Invalid GMT longitude/latitude")
                    if not current["rings"]:
                        current["rings"].append([])
                        current["ring_roles"].append("unspecified")
                    current["rings"][-1].append(point)
            except (ValueError, csv.Error) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    out = []
    for feature in features:
        rings, roles = [], []
        for points, role in zip(feature["rings"], feature["ring_roles"]):
            if not points:  # '>' before the next @D is a separator, not a lost ring.
                continue
            array = np.asarray(points, dtype=float)
            keep = np.r_[True, np.any(np.diff(array, axis=0) != 0, axis=1)]
            array = array[keep]
            if len(array) > 1 and np.array_equal(array[0], array[-1]):
                array = array[:-1]
            if len(array) < 3:
                raise ValueError(f"{path}: Ring needs three distinct vertices")
            rings.append(array)
            roles.append(role)
        if not rings:
            raise ValueError(f"{path}: @D feature has no polygon coordinates")
        meta = feature["meta"]
        out.append(
            {
                "meta": meta,
                "name": meta.get("NAME", ""),
                "type": meta.get("TYPE", ""),
                "gpgim_type": meta.get("GPGIM_TYPE", ""),
                "feature_id": meta.get("FEATURE_ID", ""),
                "plate_id": meta.get("PLATEID1", ""),
                "valid_time": (
                    meta.get("FROMAGE", meta.get("VALID_AGE", "")),
                    meta.get("TOAGE", meta.get("VALID_AGE2", "")),
                ),
                "lon": rings[0][:, 0],
                "lat": rings[0][:, 1],
                "rings": rings,
                "ring_roles": roles,
            }
        )
    return {"fields": fields, "features": out, "geometry_type": geometry_type}


def polygon_parts(feature):
    """Group each exterior with its following holes; preserve all exteriors."""
    rings = feature["rings"]
    roles = feature.get("ring_roles", ["unspecified"] * len(rings))
    if len(roles) != len(rings):
        raise ValueError("GMT ring roles must match the rings")
    if len(rings) == 1 and roles == ["unspecified"]:
        roles = ["exterior"]
    parts = []
    for ring, role in zip(rings, roles):
        if role == "exterior":
            parts.append((ring, []))
        elif role == "hole" and parts:
            parts[-1][1].append(ring)
        else:
            raise ValueError("GMT multiparts require explicit @P exteriors and following @H holes")
    return parts


def parse_time_from_name(path):
    match = re.search(r"(?<![\d.\-])(\d+(?:\.\d+)?)Ma\.gmt$", Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse snapshot age from {path}")
    return float(match.group(1))


def snapshot_paths(directory, ages):
    """Find exact requested ages; no nearest-age fallback or extrapolation."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"GMT snapshot directory not found: {directory}")
    indexed = {}
    for path in sorted(directory.glob("*.gmt")):
        age = parse_time_from_name(path)
        if age in indexed:
            raise ValueError(f"Duplicate GMT age {age:g} Ma in {directory}")
        indexed[age] = path
    selected = {}
    for age in ages:
        age = float(age)
        if not 0 <= age <= 60:
            raise ValueError("GMT geometry is restricted to 0–60 Ma")
        matches = [key for key in indexed if abs(key - age) <= 1e-6]
        if len(matches) != 1:
            raise ValueError(f"Need exactly one GMT snapshot at {age:g} Ma in {directory}")
        selected[age] = indexed[matches[0]]
    return selected


def load_snapshot(directory, time_ma):
    return parse_gmt(snapshot_paths(directory, [time_ma])[float(time_ma)])
