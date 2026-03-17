"""
geo_utils.py
------------
Utilities for loading CityFlow calibration files and projecting pixel
coordinates to geographic (world) coordinates via homography.

CityFlow calibration format
===========================
Each camera folder contains a ``calibration.txt`` with one relevant line::

    Homography matrix: h11 h12 h13;h21 h22 h23;h31 h32 h33

The stored matrix maps GPS → pixel.  ``load_homography`` returns its inverse
(pixel → GPS) so that ``pixel_to_geo`` can project detections to world coords::

    H_px2gps = inv(H_gps2px)
    [gx, gy, gw] = H_px2gps @ [u, v, 1]^T
    gps_x, gps_y = gx / gw, gy / gw

The output coordinates are GPS (latitude, longitude) in decimal degrees.
Haversine distance is used for metric distance computation.
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple


def load_homography(calibration_path: str) -> Optional[np.ndarray]:
    """
    Parse a CityFlow ``calibration.txt`` and return the 3×3 homography
    matrix in the **pixel → GPS** direction.

    The file stores the GPS→pixel homography, so this function inverts it
    so that callers can project pixel coordinates to GPS coordinates.

    Returns ``None`` if the file is missing or the line cannot be parsed.
    """
    if not calibration_path or not os.path.exists(calibration_path):
        return None
    try:
        with open(calibration_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Homography matrix:"):
                    vals_str = line[len("Homography matrix:"):].strip()
                    rows = vals_str.split(";")
                    H_gps_to_px = np.array(
                        [[float(v) for v in row.split()] for row in rows],
                        dtype=np.float64,
                    )
                    if H_gps_to_px.shape == (3, 3):
                        return np.linalg.inv(H_gps_to_px)   # pixel → GPS
    except Exception:
        pass
    return None


def pixel_to_geo(
    H: Optional[np.ndarray],
    u: float,
    v: float,
) -> Optional[Tuple[float, float]]:
    """
    Project pixel (u, v) to world coordinates using homography H.

    Returns ``(geo_x, geo_y)`` or ``None`` if H is unavailable or the
    perspective divisor is near zero.
    """
    if H is None:
        return None
    p = np.array([u, v, 1.0], dtype=np.float64)
    g = H @ p
    if abs(g[2]) < 1e-9:
        return None
    return (g[0] / g[2], g[1] / g[2])


def geo_distance(
    geo_a: Optional[Tuple[float, float]],
    geo_b: Optional[Tuple[float, float]],
) -> float:
    """
    Haversine distance in metres between two GPS points (lat, lon) in degrees.

    Returns ``float('inf')`` if either point is None.
    """
    if geo_a is None or geo_b is None:
        return float("inf")
    R = 6_371_000.0   # Earth radius in metres
    lat1, lon1 = np.radians(geo_a[0]), np.radians(geo_a[1])
    lat2, lon2 = np.radians(geo_b[0]), np.radians(geo_b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def load_timestamps(timestamp_path: str) -> Dict[str, float]:
    """
    Parse a CityFlow ``cam_timestamp/<scenario>.txt`` file.

    Format (one camera per line)::

        c001 0
        c002 1.640
        c003 2.049

    Returns a dict ``{cam_id: start_time_seconds}``.
    Returns an empty dict if the file is missing.
    """
    offsets: Dict[str, float] = {}
    if not timestamp_path or not os.path.exists(timestamp_path):
        return offsets
    with open(timestamp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                offsets[parts[0]] = float(parts[1])
    return offsets


def bbox_foot_point(bbox) -> Tuple[float, float]:
    """
    Return the bottom-centre pixel of a bounding box [x, y, w, h].

    The foot-point (ground contact) gives the best correspondence with the
    ground plane that the homography is calibrated against.
    """
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (x + w / 2.0, y + h)
