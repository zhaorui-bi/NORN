"""GPML/GPMLZ polygon parsing with the Python standard library only.

.gpmlz = gzip-compressed GPML (XML). Extracts feature polygons with
- plate id (gpml:plateid),
- name/description when present,
- exterior + interior rings (gml:posList, "lon lat lon lat ..."),
- valid time (gml:begin/endPosition) when present.

Rings keep their listed order; the exterior ring is used for plate
assignment with the validated spherical winding test in geometry.sphere.
"""

import gzip
import xml.etree.ElementTree as ET

import numpy as np


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def load_gpml(path):
    opener = gzip.open if str(path).endswith(".gpmlz") else open
    with opener(path, "rb") as f:
        tree = ET.parse(f)
    features = []
    for member in tree.iter():
        if _local(member.tag) != "featureMember":
            continue
        # the feature is a TYPED child (gpml:Basin, gpml:ClosedContinentalBoundary, ...)
        feat_elem = None
        for child in member:
            if _local(child.tag) not in ("boundingBox", "description"):
                feat_elem = child
                break
        if feat_elem is None:
            continue
        feat = {
            "feature_type": _local(feat_elem.tag),
            "plate_id": None,
            "name": "",
            "begin": None,
            "end": None,
            "rings": [],
        }
        for node in feat_elem.iter():
            t = _local(node.tag)
            if t in ("plateid", "reconstructionPlateId") and feat["plate_id"] is None:
                # value may live in a ConstantValue/value child, not in the text
                cand = (node.text or "").strip()
                if not cand:
                    for sub in node.iter():
                        if _local(sub.tag) in ("value", "plateid") and (sub.text or "").strip():
                            cand = (sub.text or "").strip()
                            break
                try:
                    feat["plate_id"] = int(float(cand))
                except ValueError:
                    pass
            elif t == "name" and not feat["name"]:
                feat["name"] = (node.text or "").strip()
            elif t == "beginPosition" and feat["begin"] is None:
                feat["begin"] = (node.text or "").strip()
            elif t == "endPosition" and feat["end"] is None:
                feat["end"] = (node.text or "").strip()
            elif t == "posList":
                vals = (node.text or "").split()
                coords = np.array(vals, dtype=float).reshape(-1, 2)
                if len(coords) >= 3:
                    feat["rings"].append(coords)
        if feat["rings"] and feat["plate_id"] is not None:
            features.append(feat)
    return features


def outer_rings(features):
    """[(plate_id, lon_array, lat_array)] using each feature's first ring."""
    out = []
    for f in features:
        ring = f["rings"][0]
        out.append((f["plate_id"], ring[:, 0], ring[:, 1]))
    return out
