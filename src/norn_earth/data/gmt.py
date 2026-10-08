"""GMT snapshot parsing (polygons + metadata), dateline/pole aware.

Parses the plate-boundary / deforming-network exports in this workspace.
Structure facts handled here:
- one GMT feature (@D record) can contain MULTIPLE rings separated by '>'
  (outer boundary + interior holes); rings are kept separate;
- rings are CLOSED (first vertex == last vertex); the duplicate closing
  vertex is dropped so area/winding math sees a clean open ring;
- metadata (NAME/TYPE/GPGIM_TYPE/PLATEID...) is identifier-only information:
  never a physical property, never a thickness label (§1.4).

These files carry closed plate polygons and TopologicalNetwork outlines --
not classified ridge/trench lines, velocities or internal strain.
"""

import re

import numpy as np


def parse_gmt(path):
    """Return {'fields': [...], 'features': [ {meta..., lon, lat, rings} ]}.

    lon/lat expose the FIRST (outer) ring for convenience; `rings` holds all
    rings as lists of (lon, lat) arrays with the closing duplicate removed.
    """
    features, cur, fields = [], None, []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if s.startswith("# @N"):
                fields = s[5:].split("|")
            elif s.startswith("# @D"):
                vals = s[5:].split("|")
                meta = dict(zip(fields, vals))
                cur = {"meta": meta, "rings": [], "_ring": None}
                features.append(cur)
            elif s.startswith(">"):
                if cur is not None:
                    cur["_ring"] = []
                    cur["rings"].append(cur["_ring"])
            elif s and not s.startswith("#"):
                parts = s.split()
                if len(parts) >= 2 and cur is not None:
                    if cur["_ring"] is None:
                        cur["_ring"] = []
                        cur["rings"].append(cur["_ring"])
                    try:
                        cur["_ring"].append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    out = []
    for f_ in features:
        rings = []
        for ring in f_["rings"]:
            if len(ring) < 3:
                continue
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue
            arr = np.array(ring, dtype=float)
            rings.append(arr)
        if not rings:
            continue
        meta = f_.get("meta", {})
        out.append(
            {
                "name": meta.get("NAME", ""),
                "type": meta.get("TYPE", ""),
                "gpgim_type": meta.get("GPGIM_TYPE", ""),
                "plate_id": meta.get("PLATEID1", ""),
                "valid_time": (meta.get("VALID_AGE", ""), meta.get("VALID_AGE2", "")),
                "lon": rings[0][:, 0],
                "lat": rings[0][:, 1],
                "rings": rings,
            }
        )
    return {"fields": fields, "features": out}


def parse_time_from_name(path):
    m = re.search(r"(\d+(?:\.\d+)?)Ma", str(path))
    if not m:
        raise ValueError(f"cannot parse time from {path}")
    return float(m.group(1))


def load_snapshot(directory, time_ma):
    directory = str(directory).rstrip("/")
    path = f"{directory}/topology_{float(time_ma):.2f}Ma.gmt"
    return parse_gmt(path)
