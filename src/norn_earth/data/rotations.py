"""GPlates rotation-file parsing and plate-circuit reconstruction (no pygplates).

.rot format (plain text): each data line is
    plate_id  age  lat  lon  angle  [conjugate_id]  [comment]
Entries give the TOTAL finite rotation of plate_id RELATIVE TO conjugate_id
(blank/0 conjugate = anchor) at that reconstruction time. Positions are
reconstructed as p(t) = R_total(plate, t) @ p(present-day), with
R_total composed along the plate circuit to the anchor and SLERP
interpolation between bracketing sampled times.

This parser supports the NORN G1 compatibility tests; production use with
deforming networks still requires pygplates + the original topologies.
"""

import math

import numpy as np

from ..geometry.sphere import lonlat_to_vectors, rotation_matrix, vectors_to_lonlat


def parse_rotations(path):
    """Return {(plate_id, age): (lat, lon, angle_deg, conjugate_id)}."""
    entries = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("!") or s.startswith("#"):
                continue
            parts = s.split()
            try:
                pid = int(parts[0])
                age = float(parts[1])
                lat, lon, ang = map(float, parts[2:5])
            except (ValueError, IndexError):
                continue
            conj = 0
            if len(parts) >= 6:
                try:
                    conj = int(parts[5])
                except ValueError:
                    conj = 0
            key = (pid, age)
            if key in entries:
                continue
            entries[key] = (lat, lon, ang, conj)
    return entries


class RotationModel:
    def __init__(self, entries, anchor=0):
        self.anchor = anchor
        # per plate: sorted list of (age, rotation_matrix, conjugate)
        by_plate = {}
        for (pid, age), (lat, lon, ang, conj) in entries.items():
            R = rotation_matrix(_axis_from_pole(lat, lon), ang) if abs(ang) > 1e-12 else np.eye(3)
            by_plate.setdefault(pid, []).append((age, R, conj))
        self.plates = {pid: sorted(vals, key=lambda x: x[0]) for pid, vals in by_plate.items()}

    def parent(self, pid, age):
        vals = self.plates.get(pid)
        if not vals:
            return None
        # conjugate at the oldest entry bracketing/covering `age` (constant in practice)
        return vals[0][2]

    def _interp(self, pid, age):
        vals = self.plates.get(pid)
        if not vals:
            return None, None
        if age <= vals[0][0]:
            return vals[0][1], vals[0][2]
        if age >= vals[-1][0]:
            return vals[-1][1], vals[-1][2]
        for (a0, R0, c0), (a1, R1, c1) in zip(vals[:-1], vals[1:]):
            if a0 <= age <= a1:
                if a1 - a0 < 1e-9:
                    return R0, c0
                # relative rotation between samples for SLERP: R_rel = R1 @ R0.T
                R_rel = R1 @ R0.T
                pole, angle = _matrix_to_pole_angle(R_rel)
                f = (age - a0) / (a1 - a0)
                R = rotation_matrix(pole, angle * f) @ R0
                return R, c0
        return vals[-1][1], vals[-1][2]

    def total_rotation(self, pid, age, max_depth=12):
        """R_total mapping present-day -> time `age` for plate pid."""
        R_total = np.eye(3)
        current = pid
        for _ in range(max_depth):
            if current == self.anchor:
                return R_total
            R_step, parent = self._interp(current, age)
            if R_step is None or parent is None or parent == current:
                return None
            R_total = R_step @ R_total
            current = parent
        return None

    def reconstruct(self, lon, lat, pid, age):
        R = self.total_rotation(pid, age)
        if R is None:
            return None
        v = lonlat_to_vectors(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        lon_r, lat_r = vectors_to_lonlat(v @ R.T)
        return lon_r, lat_r


def _axis_from_pole(lat, lon):
    return lonlat_to_vectors(lon, lat)


def _matrix_to_pole_angle(R):
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    w, v = np.linalg.eig(R)
    axis = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    axis = axis / np.linalg.norm(axis)
    if np.trace(R @ _cross_matrix(axis) - _cross_matrix(axis) @ R) < 0:
        pass
    # sign via skew part
    skew = (R - R.T) / 2.0
    axis2 = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if np.dot(axis2, axis) < 0:
        axis = -axis
    return axis, math.degrees(angle)


def _cross_matrix(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
