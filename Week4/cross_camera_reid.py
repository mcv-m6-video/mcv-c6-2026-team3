"""
cross_camera_reid.py
--------------------
Cross-camera Re-Identification module for MTMC (Multi-Target Multi-Camera) tracking.

Architecture
============
GlobalIDManager   – single source of truth for all track IDs across every camera.
AppearanceBuffer  – per-camera ring-buffer that stores recent
                    (frame_id, track_id, crop, geo_pos) tuples so other cameras
                    can query "who was seen here recently?".
CrossCameraReIDMatcher – orchestrates matching between a new detection in camera A and
                         all recently-seen tracks in every other camera.  The scoring
                         function is fully pluggable (swap histogram → deep ReID, etc.).

Geographic enhancement
======================
When camera homographies and FPS are provided, matching incorporates:
  * Speed gating  – candidate tracks whose implied travel speed exceeds
                    `max_speed` (world-coord units / second) are rejected.
  * Geo score     – a distance-decay term exp(-dist/geo_sigma) that rewards
                    spatially proximate candidates.
  * Combined score = (1 - geo_weight) * appearance + geo_weight * geo_score

Pluggable scorers
=================
Any callable with signature
    scorer(crop_a: np.ndarray, crop_b: np.ndarray) -> float   (higher = more similar)
can be passed as `scorer_fn` to CrossCameraReIDMatcher.

Built-in scorers shipped here:
    histogram_scorer   – fast BGR histogram correlation  (default)
    You can add your own and pass it at construction time.
"""

import cv2
import numpy as np
from collections import deque
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

from geo_utils import geo_distance


# ---------------------------------------------------------------------------
# Scorer interface & built-in implementations
# ---------------------------------------------------------------------------

def histogram_scorer(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """
    Compare two BGR image crops using normalised histogram correlation.

    Returns a value in [-1, 1]; 1.0 = identical distribution.
    Typically use a threshold around 0.5–0.7 in practice.
    """
    if crop_a is None or crop_b is None:
        return 0.0
    if crop_a.size == 0 or crop_b.size == 0:
        return 0.0

    score = 0.0
    for ch in range(3):                         # B, G, R channels
        h_a = cv2.calcHist([crop_a], [ch], None, [32], [0, 256])
        h_b = cv2.calcHist([crop_b], [ch], None, [32], [0, 256])
        cv2.normalize(h_a, h_a)
        cv2.normalize(h_b, h_b)
        score += cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL)

    return score / 3.0                          # average over channels


def dummy_scorer(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """
    Placeholder scorer that always returns 0 (no match).
    Useful when you have no frame/crop available and only want Kalman tracking.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Global ID Manager
# ---------------------------------------------------------------------------

class GlobalIDManager:
    """
    Thread-safe global counter that issues unique track IDs across all cameras.

    Every SORTTracker should call `new_id()` instead of using its own counter
    so that IDs are never reused across cameras.

    Usage
    -----
        gid = GlobalIDManager()
        id1 = gid.new_id()   # → 1
        id2 = gid.new_id()   # → 2
        ...
    """

    def __init__(self, start: int = 1):
        self._counter = start - 1
        self._lock = Lock()

    def new_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    @property
    def current(self) -> int:
        return self._counter

    def reset(self, start: int = 1):
        with self._lock:
            self._counter = start - 1


# ---------------------------------------------------------------------------
# Appearance Buffer (per camera)
# ---------------------------------------------------------------------------

class AppearanceBuffer:
    """
    Stores recent (abs_time, global_track_id, crop, geo_pos) entries for ONE camera.

    ``abs_time`` is wall-clock seconds from scenario start, already accounting
    for the camera's timestamp offset.  Entries older than ``max_seconds`` are
    evicted.

    Parameters
    ----------
    camera_id   : str   – human-readable label, e.g. "c001"
    max_seconds : float – how many seconds of history to retain
    """

    def __init__(self, camera_id: str, max_seconds: float = 3.0):
        self.camera_id  = camera_id
        self.max_seconds = max_seconds
        # deque of (abs_time, global_track_id, crop_ndarray, geo_pos)
        # geo_pos is (float, float) | None
        self._entries: deque = deque()
        self._lock = Lock()

    def add(
        self,
        abs_time: float,
        global_track_id: int,
        crop: np.ndarray,
        geo_pos=None,
    ):
        """Add a crop (and optional GPS position) for a track at ``abs_time``."""
        with self._lock:
            self._entries.append((abs_time, global_track_id, crop, geo_pos))
            self._evict()

    def _evict(self):
        """Remove entries older than ``max_seconds`` relative to the latest entry."""
        if not self._entries:
            return
        latest = self._entries[-1][0]
        cutoff = latest - self.max_seconds
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def get_recent(self) -> List[Tuple]:
        """Return list of (abs_time, global_track_id, crop, geo_pos)."""
        with self._lock:
            return list(self._entries)

    def get_tracks(self) -> Dict[int, Tuple]:
        """
        Return {global_track_id: (crop, abs_time, geo_pos)} for all tracks
        currently in the buffer.  If a track has multiple entries, the newest
        one is kept.
        """
        with self._lock:
            result: Dict[int, Tuple] = {}
            for abs_time, tid, crop, geo_pos in self._entries:
                result[tid] = (crop, abs_time, geo_pos)
            return result


# ---------------------------------------------------------------------------
# Cross-Camera ReID Matcher
# ---------------------------------------------------------------------------

class CrossCameraReIDMatcher:
    """
    Matches a new detection (crop) in one camera against recently-seen tracks
    in all *other* cameras using a pluggable appearance scorer.

    All timing uses **absolute wall-clock seconds** (scenario time) so that
    cameras with different start offsets are compared correctly.  Each
    ``SORTTracker`` is responsible for computing
    ``abs_time = timestamp_offset + frame_id / fps``
    before calling ``update_appearance`` or ``match_or_create``.

    Parameters
    ----------
    global_id_mgr     : GlobalIDManager
    scorer_fn         : Callable[[np.ndarray, np.ndarray], float]
    match_threshold   : float   – minimum combined score to accept a match
    lookback_seconds  : float   – seconds of history to search per camera

    Geographic parameters (geo scoring disabled when geo_weight=0)
    ---------------------------------------------------------------
    max_speed         : float  – max plausible speed in m/s (hard gate ≈108 km/h)
    geo_sigma         : float  – distance-decay constant in metres
    geo_weight        : float  – blend weight [0 = appearance only, 1 = geo only]
    """

    def __init__(
        self,
        global_id_mgr: GlobalIDManager,
        scorer_fn: Callable[[np.ndarray, np.ndarray], float] = histogram_scorer,
        match_threshold: float = 0.6,
        lookback_seconds: float = 3.0,
        # --- geographic parameters ---
        max_speed: float = 30.0,   # metres/sec (~108 km/h hard gate)
        geo_sigma: float = 50.0,   # metres; distance-decay constant (exp(-d/sigma))
        geo_weight: float = 0.3,
    ):
        self.global_id_mgr   = global_id_mgr
        self.scorer_fn       = scorer_fn
        self.match_threshold = match_threshold
        self.lookback_seconds = lookback_seconds
        self.max_speed       = max_speed
        self.geo_sigma       = geo_sigma
        self.geo_weight      = geo_weight

        self._buffers: Dict[str, AppearanceBuffer] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_camera(self, camera_id: str):
        """Create an AppearanceBuffer for `camera_id` if it doesn't exist."""
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = AppearanceBuffer(
                    camera_id, max_seconds=self.lookback_seconds
                )

    def update_appearance(
        self,
        camera_id: str,
        abs_time: float,
        global_track_id: int,
        crop: Optional[np.ndarray],
        geo_pos=None,
    ):
        """
        Record that ``global_track_id`` was seen in ``camera_id`` at scenario
        time ``abs_time`` (seconds) with appearance ``crop`` and GPS ``geo_pos``.
        """
        if camera_id not in self._buffers:
            self.register_camera(camera_id)
        if crop is not None and crop.size > 0:
            self._buffers[camera_id].add(abs_time, global_track_id, crop, geo_pos)

    def match_or_create(
        self,
        camera_id: str,
        abs_time: float,
        crop: Optional[np.ndarray],
        geo_pos=None,
    ) -> int:
        """
        Try to match a new detection against tracks in other cameras.

        Parameters
        ----------
        camera_id : source camera
        abs_time  : scenario time in seconds (offset + frame_id / fps)
        crop      : appearance crop (BGR array or None)
        geo_pos   : GPS (lat, lon) of the detection's foot-point, or None

        Returns
        -------
        int – existing global ID (cross-camera match) or a fresh one.
        """
        if camera_id not in self._buffers:
            self.register_camera(camera_id)

        best_id, best_score = self._search_other_cameras(
            camera_id, abs_time, crop, geo_pos
        )

        if best_id is not None and best_score >= self.match_threshold:
            return best_id
        else:
            return self.global_id_mgr.new_id()

    def swap_scorer(
        self,
        new_scorer_fn: Callable[[np.ndarray, np.ndarray], float],
        new_threshold: Optional[float] = None,
    ):
        """
        Hot-swap the appearance scorer at runtime.

        Parameters
        ----------
        new_scorer_fn  : new callable (crop_a, crop_b) -> float
        new_threshold  : optionally update the match threshold too
        """
        self.scorer_fn = new_scorer_fn
        if new_threshold is not None:
            self.match_threshold = new_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_other_cameras(
        self,
        source_camera_id: str,
        query_abs_time: float,
        query_crop: Optional[np.ndarray],
        query_geo=None,
    ) -> Tuple[Optional[int], float]:
        """
        Score every recently-seen track in other cameras against the query.

        ``query_abs_time`` is scenario wall-clock seconds, already accounting
        for the source camera's timestamp offset, so cross-camera dt is exact.

        Combined score
        --------------
        When geo data is available::

            dt          = |query_abs_time - gallery_abs_time|   (seconds)
            speed       = haversine_dist / dt                   (m/s)
            speed_gate  : reject if speed > max_speed
            speed_score = 1 - speed / max_speed
            dist_score  = exp(-dist / geo_sigma)
            geo_score   = 0.5 * speed_score + 0.5 * dist_score
            combined    = (1-w)*appearance + w*geo_score

        Falls back to pure appearance when geo data is unavailable.
        """
        best_id: Optional[int] = None
        best_score: float = -np.inf

        for cam_id, buffer in self._buffers.items():
            if cam_id == source_camera_id:
                continue

            for tid, (gallery_crop, gallery_abs_time, gallery_geo) in buffer.get_tracks().items():
                # --- Appearance score ---
                app_score = self.scorer_fn(query_crop, gallery_crop)

                # --- Geo score (optional) ---
                if self.geo_weight > 0 and query_geo is not None and gallery_geo is not None:
                    dist = geo_distance(query_geo, gallery_geo)
                    dt   = abs(query_abs_time - gallery_abs_time)

                    if dt > 0:
                        speed = dist / dt
                        if speed > self.max_speed:
                            continue          # physically impossible – reject
                        speed_score = 1.0 - speed / self.max_speed
                    else:
                        speed_score = 1.0 if dist < 1.0 else 0.0

                    dist_score = float(np.exp(-dist / max(self.geo_sigma, 1e-6)))
                    geo_score  = 0.5 * speed_score + 0.5 * dist_score
                    combined   = (1.0 - self.geo_weight) * app_score + self.geo_weight * geo_score
                else:
                    combined = app_score

                if combined > best_score:
                    best_score = combined
                    best_id    = tid

        return best_id, best_score
