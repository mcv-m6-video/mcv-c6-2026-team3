"""
sort_tracker.py
---------------
SORT tracker extended with optional cross-camera Re-ID support.

Changes vs. original
====================
* KalmanBoxTracker no longer uses its own class-level counter.
  IDs are now issued externally (either by GlobalIDManager for MTMC runs,
  or by a simple local counter for single-camera backwards-compatible runs).

* SORTTracker accepts three new optional constructor arguments:
    - reid_matcher   : CrossCameraReIDMatcher | None
    - camera_id      : str | None
    - global_id_mgr  : GlobalIDManager | None

  When `reid_matcher` is None the tracker behaves exactly as before.

* SORTTracker.track() now accepts an optional `frame` argument (the raw
  BGR numpy array) so it can extract crops for ReID.  Passing `frame=None`
  is always safe.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from utils import compute_iou
from geo_utils import pixel_to_geo, bbox_foot_point


# ---------------------------------------------------------------------------
# Kalman Box Tracker
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Tracks a single bounding box with a constant-velocity Kalman filter.

    State vector: [x_center, y_center, scale, aspect_ratio, vx, vy, vs]
    Measurement:  [x_center, y_center, scale, aspect_ratio]
    """

    def __init__(self, bbox, track_id: int):
        """
        Parameters
        ----------
        bbox     : [x, y, w, h]
        track_id : externally assigned unique integer ID
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.bbox_to_z(bbox)

        self.time_since_update = 0
        self.id                = track_id   # globally unique ID
        self.history           = []
        self.hits              = 0
        self.hit_streak        = 0
        self.age               = 0

    # ------------------------------------------------------------------
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.z_to_bbox(self.kf.x)

    # ------------------------------------------------------------------
    @staticmethod
    def bbox_to_z(bbox):
        x, y, w, h = bbox[:4]
        x_c = x + w / 2.0
        y_c = y + h / 2.0
        s   = w * h
        r   = w / float(h) if h != 0 else 1.0
        return np.array([x_c, y_c, s, r]).reshape((4, 1))

    @staticmethod
    def z_to_bbox(x):
        x_c, y_c, s, r = x[0], x[1], x[2], x[3]
        w = np.sqrt(s * r)
        h = s / w if w != 0 else 1.0
        return np.array([x_c - w / 2.0, y_c - h / 2.0, w, h]).flatten()


# ---------------------------------------------------------------------------
# SORT Tracker (with optional MTMC ReID)
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    SORT tracker with optional cross-camera Re-ID.

    Single-camera mode (backward compatible)
    -----------------------------------------
    tracker = SORTTracker(max_age=5, min_hits=3, iou_threshold=0.3)
    results = tracker.track(detections, frame_id)

    MTMC mode
    ---------
    from cross_camera_reid import CrossCameraReIDMatcher, GlobalIDManager

    gid   = GlobalIDManager()
    reid  = CrossCameraReIDMatcher(gid, scorer_fn=histogram_scorer,
                                   match_threshold=0.6, lookback_frames=30)

    tracker_c001 = SORTTracker(..., reid_matcher=reid,
                                    camera_id="c001",
                                    global_id_mgr=gid)
    tracker_c002 = SORTTracker(..., reid_matcher=reid,
                                    camera_id="c002",
                                    global_id_mgr=gid)

    # Pass the raw frame so crops can be extracted for ReID
    results = tracker_c001.track(detections, frame_id, frame=bgr_frame)
    """

    def __init__(
        self,
        max_age:       int   = 1,
        min_hits:      int   = 3,
        iou_threshold: float = 0.3,
        # --- MTMC extensions (all optional) ---
        reid_matcher=None,              # CrossCameraReIDMatcher | None
        camera_id:         Optional[str]           = None,
        global_id_mgr=None,             # GlobalIDManager | None
        homography:        Optional[np.ndarray]    = None,   # 3×3 pixel→GPS matrix
        fps:               float                   = 10.0,   # camera frame rate
        timestamp_offset:  float                   = 0.0,    # seconds; camera start time
    ):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold

        # MTMC components
        self.reid_matcher     = reid_matcher
        self.camera_id        = camera_id
        self.global_id_mgr    = global_id_mgr
        self.homography       = homography
        self.fps              = fps
        self.timestamp_offset = timestamp_offset

        # Register this camera with the matcher
        if self.reid_matcher is not None and self.camera_id is not None:
            self.reid_matcher.register_camera(self.camera_id)

        # Local fallback counter (used only when global_id_mgr is None)
        self._local_id_counter = 0

        self.trackers:    List[KalmanBoxTracker] = []
        self.frame_count: int = 0
        self.tracks:      Dict[int, List[Tuple]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        self.trackers    = []
        self.frame_count = 0
        self.tracks      = {}
        self._local_id_counter = 0

    def track(
        self,
        detections: List[Tuple],
        frame_id:   int,
        frame:      Optional[np.ndarray] = None,   # BGR image for crop extraction
    ) -> List[Tuple]:
        """
        Parameters
        ----------
        detections : list of [x, y, w, h, ...]
        frame_id   : integer frame index
        frame      : optional BGR numpy array – required for appearance-based ReID

        Returns
        -------
        list of (global_track_id, (x, y, w, h))
        """
        self.frame_count += 1

        # ---- Kalman predict ----
        trks    = np.zeros((len(self.trackers), 4))
        to_del  = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets = np.array([d[:4] for d in detections]) if detections else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = \
            self._associate_detections_to_trackers(dets, trks)

        abs_t = self._abs_time(frame_id)

        # ---- Update matched trackers ----
        for m in matched:
            trk_obj  = self.trackers[m[0]]
            det_bbox = dets[m[1], :]
            trk_obj.update(det_bbox)

            # Push appearance + geo update to ReID buffer
            if self.reid_matcher is not None and frame is not None:
                crop    = self._extract_crop(frame, det_bbox)
                geo_pos = self._bbox_to_geo(det_bbox)
                self.reid_matcher.update_appearance(
                    self.camera_id, abs_t, trk_obj.id, crop, geo_pos=geo_pos
                )

        # ---- Create new trackers for unmatched detections ----
        for i in unmatched_dets:
            det_bbox  = dets[i, :]
            crop      = self._extract_crop(frame, det_bbox) if frame is not None else None
            geo_pos   = self._bbox_to_geo(det_bbox)

            new_id = self._assign_new_id(crop, abs_t, geo_pos=geo_pos)
            trk    = KalmanBoxTracker(det_bbox, track_id=new_id)
            self.trackers.append(trk)

            # Immediately add to ReID buffer so subsequent cameras can match
            if self.reid_matcher is not None and crop is not None:
                self.reid_matcher.update_appearance(
                    self.camera_id, abs_t, new_id, crop, geo_pos=geo_pos
                )

        # ---- Collect output & prune dead tracks ----
        tracked_objects = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()

            visible = (
                trk.time_since_update < 1
                and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            )
            if visible:
                x, y, w, h = d
                tracked_objects.append((trk.id, (int(x), int(y), int(w), int(h))))

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        self.tracks[frame_id] = tracked_objects
        return tracked_objects

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _abs_time(self, frame_id: int) -> float:
        """Convert a frame index to absolute scenario time in seconds."""
        return self.timestamp_offset + frame_id / self.fps

    def _bbox_to_geo(self, bbox: np.ndarray):
        """
        Project a bounding box's foot-point (bottom-centre pixel) to world
        coordinates using this camera's homography.  Returns None if no
        homography is set or projection fails.
        """
        if self.homography is None:
            return None
        u, v = bbox_foot_point(bbox)
        return pixel_to_geo(self.homography, u, v)

    def _assign_new_id(
        self,
        crop:     Optional[np.ndarray],
        abs_time: float,
        geo_pos=None,
    ) -> int:
        """
        Attempt cross-camera ReID; fall back to a fresh ID if no matcher
        is configured or no match is found above threshold.
        """
        if self.reid_matcher is not None and self.camera_id is not None:
            active_ids = {trk.id for trk in self.trackers}
            proposed = self.reid_matcher.match_or_create(
                camera_id=self.camera_id,
                abs_time=abs_time,
                crop=crop,
                geo_pos=geo_pos,
            )
            # Reject proposals already active in this camera to avoid
            # assigning the same global ID to two different local tracks.
            if proposed not in active_ids:
                return proposed
            return self.global_id_mgr.new_id()

        # Single-camera fallback: use global manager if provided, else local
        if self.global_id_mgr is not None:
            return self.global_id_mgr.new_id()

        self._local_id_counter += 1
        return self._local_id_counter

    @staticmethod
    def _extract_crop(
        frame: np.ndarray,
        bbox:  np.ndarray,
    ) -> Optional[np.ndarray]:
        """Safely extract a BGR crop from `frame` given [x, y, w, h] bbox."""
        if frame is None:
            return None
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        H, W = frame.shape[:2]
        x1 = max(0, x);     y1 = max(0, y)
        x2 = min(W, x + w); y2 = min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers:   np.ndarray,
    ):
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int),
            )
        if len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=int),
                np.arange(len(trackers)),
            )

        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            trk_xyxy = (trk[0], trk[1], trk[0] + trk[2], trk[1] + trk[3])
            for d, det in enumerate(detections):
                det_xyxy = (det[0], det[1], det[0] + det[2], det[1] + det[3])
                iou_matrix[t, d] = compute_iou(trk_xyxy, det_xyxy)

        row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
        matched_indices  = np.stack([row_ind, col_ind], axis=1)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                matches.append(m.reshape(1, 2))

        if matches:
            matches = np.concatenate(matches, axis=0)
        else:
            matches = np.empty((0, 2), dtype=int)

        unmatched_dets = [
            d for d in range(len(detections))
            if len(matches) == 0 or d not in matches[:, 1]
        ]
        unmatched_trks = [
            t for t in range(len(trackers))
            if len(matches) == 0 or t not in matches[:, 0]
        ]

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# ---------------------------------------------------------------------------
# ByteTrack MTMC Tracker
# ---------------------------------------------------------------------------

class ByteTrackMTMCTracker:
    """
    Drop-in replacement for ``SORTTracker`` that uses ``supervision.ByteTrack``
    internally while wiring into the same cross-camera ReID / GlobalIDManager
    infrastructure.

    ByteTrack assigns its own monotonically-increasing local integer IDs.
    This class maintains a ``local_id → global_id`` mapping and calls
    ``reid_matcher.match_or_create`` the first time a local ID is seen,
    exactly as ``SORTTracker`` does for unmatched detections.

    Parameters
    ----------
    track_thresh   : float  – high-confidence detection threshold
    track_buffer   : int    – frames to keep a lost track alive
    match_thresh   : float  – IoU threshold for ByteTrack's internal matching
    fps            : float  – frame rate of this camera's video
    timestamp_offset : float – camera start time offset in seconds (scenario)
    reid_matcher   : CrossCameraReIDMatcher | None
    camera_id      : str | None
    global_id_mgr  : GlobalIDManager | None
    homography     : np.ndarray | None  – 3×3 pixel→GPS matrix
    """

    def __init__(
        self,
        track_thresh:     float = 0.25,
        track_buffer:     int   = 30,
        match_thresh:     float = 0.8,
        fps:              float = 10.0,
        timestamp_offset: float = 0.0,
        # MTMC extensions
        reid_matcher=None,
        camera_id:        Optional[str]        = None,
        global_id_mgr=None,
        homography:       Optional[np.ndarray] = None,
    ):
        from supervision import Detections, ByteTrack as _ByteTrack

        self.track_buffer     = track_buffer
        self.fps              = fps
        self.timestamp_offset = timestamp_offset
        self.reid_matcher     = reid_matcher
        self.camera_id        = camera_id
        self.global_id_mgr    = global_id_mgr
        self.homography       = homography

        self._ByteTrack  = _ByteTrack
        self._Detections = Detections

        self._bt = _ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=int(fps),
        )

        # local ByteTrack ID → global MTMC ID
        self._local_to_global: Dict[int, int] = {}
        # local ID → last frame it was seen (for stale-entry cleanup)
        self._last_seen:       Dict[int, int] = {}

        self._local_id_counter = 0
        self.frame_count = 0
        self.tracks: Dict[int, List[Tuple]] = {}

        if self.reid_matcher is not None and self.camera_id is not None:
            self.reid_matcher.register_camera(self.camera_id)

    def reset(self):
        self._bt = self._ByteTrack(
            track_activation_threshold=self._bt.track_activation_threshold
            if hasattr(self._bt, "track_activation_threshold") else 0.25,
            lost_track_buffer=self.track_buffer,
            frame_rate=int(self.fps),
        )
        self._local_to_global.clear()
        self._last_seen.clear()
        self._local_id_counter = 0
        self.frame_count = 0
        self.tracks = {}

    def track(
        self,
        detections: List[Tuple],
        frame_id:   int,
        frame:      Optional[np.ndarray] = None,
    ) -> List[Tuple]:
        """Same interface as ``SORTTracker.track``."""
        self.frame_count += 1
        abs_t = self.timestamp_offset + frame_id / self.fps

        # ---- Build supervision Detections ----
        if detections:
            xyxy_boxes   = []
            confidences  = []
            for det in detections:
                x, y, w, h = det[:4]
                conf = float(det[4]) if len(det) > 4 else 1.0
                xyxy_boxes.append([x, y, x + w, y + h])
                confidences.append(conf)
            sv_dets = self._Detections(
                xyxy=np.array(xyxy_boxes,  dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.zeros(len(xyxy_boxes), dtype=int),
            )
        else:
            sv_dets = self._Detections.empty()

        tracked = self._bt.update_with_detections(sv_dets)

        # ---- Map local → global IDs ----
        tracked_objects: List[Tuple] = []

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            # IDs currently active in this camera (global)
            active_global = set(self._local_to_global.values())

            for i, local_id in enumerate(tracked.tracker_id):
                local_id = int(local_id)
                x1, y1, x2, y2 = tracked.xyxy[i]
                bbox = np.array([x1, y1, x2 - x1, y2 - y1])
                crop    = SORTTracker._extract_crop(frame, bbox)
                geo_pos = self._bbox_to_geo(bbox)

                if local_id not in self._local_to_global:
                    # New local track — attempt cross-camera ReID
                    proposed = self._reid_match_or_create(abs_t, crop, geo_pos)
                    if proposed in active_global:
                        # Collision: another local track in this camera already
                        # holds this global ID; issue a fresh one instead.
                        proposed = self._new_local_id()
                    self._local_to_global[local_id] = proposed
                    active_global.add(proposed)

                global_id = self._local_to_global[local_id]
                self._last_seen[local_id] = frame_id

                # Push appearance update so other cameras can query it
                if self.reid_matcher is not None and crop is not None:
                    self.reid_matcher.update_appearance(
                        self.camera_id, abs_t, global_id, crop, geo_pos=geo_pos
                    )

                tracked_objects.append(
                    (global_id, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                )

        # Clean up stale local→global entries (not seen for track_buffer frames)
        stale = [lid for lid, last in self._last_seen.items()
                 if frame_id - last > self.track_buffer]
        for lid in stale:
            self._local_to_global.pop(lid, None)
            self._last_seen.pop(lid, None)

        self.tracks[frame_id] = tracked_objects
        return tracked_objects

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reid_match_or_create(self, abs_t: float, crop, geo_pos) -> int:
        if self.reid_matcher is not None and self.camera_id is not None:
            return self.reid_matcher.match_or_create(
                camera_id=self.camera_id,
                abs_time=abs_t,
                crop=crop,
                geo_pos=geo_pos,
            )
        return self._new_local_id()

    def _new_local_id(self) -> int:
        if self.global_id_mgr is not None:
            return self.global_id_mgr.new_id()
        self._local_id_counter += 1
        return self._local_id_counter

    def _bbox_to_geo(self, bbox: np.ndarray):
        if self.homography is None:
            return None
        u, v = bbox_foot_point(bbox)
        return pixel_to_geo(self.homography, u, v)