"""
offline_mtmc.py
---------------
Offline Multi-Target Multi-Camera tracking pipeline.

Steps
-----
1. Per-camera local tracking (ByteTrack or SORT, no cross-camera ReID)
2. Build tracklets with GPS positions and per-frame velocity profiles
3. Filter parked vehicles (start/end pixel displacement < threshold)
4. Cross-camera greedy matching using four signals:
     a. Temporal plausibility  (hard gate + initial sort)
     b. Geographic proximity   at overlapping/nearest absolute times
     c. Velocity similarity    at closest temporal overlap
     d. Appearance             histogram or Siamese embeddings
5. Output per-camera MOTChallenge txt files (global IDs, multi-camera tracks only)
6. Optional evaluation against GT

Usage
-----
    python offline_mtmc.py \\
        --scenario_dir .../train/S01 \\
        --dets_dir results/finetuned/S01 \\
        --use_roi --conf_thr 0.4 \\
        --eval
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse existing modules
from sort_tracker import SORTTracker
from sort_tracker import ByteTrackMTMCTracker
from geo_utils import (
    load_homography, load_timestamps,
    pixel_to_geo, geo_distance, geo_to_local_vel, predict_geo, bbox_foot_point,
)
from cross_camera_reid import histogram_scorer
from main import (
    load_detections, find_detection_file, discover_cameras,
    load_roi_mask, SiameseScorer, _build_scorer,
    render_outputs,
)

try:
    from evaluation.tracking_eval import load_gt_tracks, evaluate_tracking
except ImportError:
    load_gt_tracks = evaluate_tracking = None

try:
    from aic_eval import (
        load_aic_gt_for_sequence, merge_cam_outputs,
        run_aic_eval, print_aic_results,
    )
    _HAVE_AIC = True
except ImportError:
    _HAVE_AIC = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    frame_0idx: int
    abs_time:   float
    bbox:       np.ndarray      # [x, y, w, h]
    geo_pos:    Optional[Tuple[float, float]]
    velocity:   Optional[Tuple[float, float]]   # (v_north_m_s, v_east_m_s)
    crop:       Optional[np.ndarray]


@dataclass
class Tracklet:
    cam:      str
    local_id: int
    records:  List[TrackRecord] = field(default_factory=list)

    # Cached derived values (filled by build_tracklet_meta)
    start_time:  float = 0.0
    end_time:    float = 0.0
    start_px:    Optional[np.ndarray] = None   # bbox centre at first record
    end_px:      Optional[np.ndarray] = None   # bbox centre at last record
    mean_hist:   Optional[np.ndarray] = None
    mean_emb:    Optional[np.ndarray] = None   # Siamese embedding mean

    def px_displacement(self) -> float:
        """Pixel distance between first and last bbox centre."""
        if self.start_px is None or self.end_px is None:
            return 0.0
        return float(np.linalg.norm(self.start_px - self.end_px))

    def geo_at(self, t: float) -> Optional[Tuple[float, float]]:
        """
        GPS position at absolute time ``t`` via interpolation/extrapolation.
        Extrapolation uses the nearest endpoint velocity.
        Returns None if no GPS data is available.
        """
        gps_records = [(r.abs_time, r.geo_pos, r.velocity)
                       for r in self.records if r.geo_pos is not None]
        if not gps_records:
            return None

        times = [r[0] for r in gps_records]

        if t <= times[0]:
            geo, vel = gps_records[0][1], gps_records[0][2]
            if vel is None:
                return geo
            return predict_geo(geo, vel, t - times[0])

        if t >= times[-1]:
            geo, vel = gps_records[-1][1], gps_records[-1][2]
            if vel is None:
                return geo
            return predict_geo(geo, vel, t - times[-1])

        # Linear interpolation between the two bracketing records
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                alpha = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
                g0, g1 = gps_records[i][1], gps_records[i + 1][1]
                lat = g0[0] + alpha * (g1[0] - g0[0])
                lon = g0[1] + alpha * (g1[1] - g0[1])
                return (lat, lon)
        return None

    def vel_at(self, t: float) -> Optional[Tuple[float, float]]:
        """
        Velocity (v_north, v_east) m/s at the record closest to absolute time ``t``.
        """
        if not self.records:
            return None
        closest = min(self.records, key=lambda r: abs(r.abs_time - t))
        return closest.velocity


# ---------------------------------------------------------------------------
# Union-Find with camera-collision guard
# ---------------------------------------------------------------------------

class UnionFind:
    """
    Disjoint-set structure for tracklet groups.
    Prevents two tracklets from the same camera ending up in the same group.
    """

    def __init__(self):
        self._parent: Dict[int, int] = {}
        self._cams:   Dict[int, set] = {}

    def add(self, tid: int, cam: str):
        self._parent[tid] = tid
        self._cams[tid]   = {cam}

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def can_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        return not (self._cams[ra] & self._cams[rb])

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._cams[ra].update(self._cams[rb])
        self._parent[rb] = ra
        return True

    def groups(self) -> Dict[int, List[int]]:
        """Return {root: [member_ids]}."""
        g: Dict[int, List[int]] = defaultdict(list)
        for tid in self._parent:
            g[self.find(tid)].append(tid)
        return dict(g)


# ---------------------------------------------------------------------------
# Tracklet building
# ---------------------------------------------------------------------------

def _bbox_centre(bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0])


def build_tracklets(
    tracker_output: Dict[int, List[Tuple]],   # {frame_0idx: [(local_id, bbox)]}
    frame_to_abs:   Dict[int, float],          # frame_0idx -> abs_time
    homography:     Optional[np.ndarray],
    frames:         Dict[int, Optional[np.ndarray]],  # frame_0idx -> BGR image | None
    cam:            str,
    velocity_window: int = 5,
) -> Dict[int, Tracklet]:
    """
    Build per-local-ID Tracklets from raw tracker output.
    Computes per-frame GPS positions and a moving-average velocity profile.
    """
    raw: Dict[int, List[TrackRecord]] = defaultdict(list)

    for frame_0idx, tracks in sorted(tracker_output.items()):
        abs_t = frame_to_abs.get(frame_0idx, float(frame_0idx))
        frame_img = frames.get(frame_0idx)

        for local_id, bbox in tracks:
            bbox = np.array(bbox, dtype=float)

            # GPS foot-point
            geo_pos = None
            if homography is not None:
                u, v = bbox_foot_point(bbox)
                geo_pos = pixel_to_geo(homography, u, v)

            # Crop
            crop = None
            if frame_img is not None:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                H, W = frame_img.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if x2 > x1 and y2 > y1:
                    crop = frame_img[y1:y2, x1:x2].copy()

            raw[local_id].append(TrackRecord(
                frame_0idx=frame_0idx,
                abs_time=abs_t,
                bbox=bbox,
                geo_pos=geo_pos,
                velocity=None,
                crop=crop,
            ))

    tracklets: Dict[int, Tracklet] = {}
    for local_id, records in raw.items():
        records.sort(key=lambda r: r.abs_time)

        # Compute moving-average velocity profile
        vel_window: List[Tuple[float, float]] = []
        for i, rec in enumerate(records):
            if i > 0 and rec.geo_pos is not None and records[i - 1].geo_pos is not None:
                v = geo_to_local_vel(
                    records[i - 1].geo_pos, records[i - 1].abs_time,
                    rec.geo_pos,            rec.abs_time,
                )
                if v is not None:
                    vel_window.append(v)
                    if len(vel_window) > velocity_window:
                        vel_window.pop(0)
            if vel_window:
                mean_v = (
                    float(np.mean([v[0] for v in vel_window])),
                    float(np.mean([v[1] for v in vel_window])),
                )
                records[i] = TrackRecord(
                    frame_0idx=rec.frame_0idx, abs_time=rec.abs_time,
                    bbox=rec.bbox, geo_pos=rec.geo_pos,
                    velocity=mean_v, crop=rec.crop,
                )

        t = Tracklet(cam=cam, local_id=local_id, records=records)
        t.start_px = _bbox_centre(records[0].bbox)
        t.end_px   = _bbox_centre(records[-1].bbox)
        tracklets[local_id] = t

    return tracklets


def build_tracklet_meta(tracklet: Tracklet, scorer_fn, siamese_scorer=None, sample_every: int = 5):
    """Fill cached fields: time bounds, pixel endpoints, appearance representation."""
    recs = tracklet.records
    if not recs:
        return

    tracklet.start_time = recs[0].abs_time
    tracklet.end_time   = recs[-1].abs_time
    tracklet.start_px   = _bbox_centre(recs[0].bbox)
    tracklet.end_px     = _bbox_centre(recs[-1].bbox)

    # Appearance: mean histogram over sampled crops
    crops = [r.crop for r in recs[::sample_every] if r.crop is not None and r.crop.size > 0]
    if crops:
        if siamese_scorer is not None:
            import torch
            import torch.nn.functional as F
            embs = [siamese_scorer._embed(c) for c in crops]
            embs = [e for e in embs if e is not None]
            if embs:
                tracklet.mean_emb = np.mean(embs, axis=0)
        else:
            hists = []
            for crop in crops:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(h, h)
                hists.append(h.flatten())
            tracklet.mean_hist = np.mean(hists, axis=0)


# ---------------------------------------------------------------------------
# Parked car filter
# ---------------------------------------------------------------------------

def filter_parked(
    tracklets: Dict[int, Tracklet],
    min_displacement_px: float = 50.0,
) -> Dict[int, Tracklet]:
    """Remove tracklets whose first-to-last pixel displacement is below threshold."""
    return {lid: t for lid, t in tracklets.items()
            if t.px_displacement() >= min_displacement_px}


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def _temporal_score(ta: Tracklet, tb: Tracklet, max_time_gap: float) -> Optional[float]:
    """
    Returns a temporal plausibility score in [0,1], or None if time gap exceeds max_time_gap.
    Overlap → 1.0; gap → exp(-gap / max_time_gap).
    """
    overlap_start = max(ta.start_time, tb.start_time)
    overlap_end   = min(ta.end_time,   tb.end_time)
    if overlap_end >= overlap_start:
        return 1.0   # tracks overlap in time

    gap = max(ta.start_time - tb.end_time, tb.start_time - ta.end_time)
    if gap > max_time_gap:
        return None
    return float(np.exp(-gap / max(max_time_gap, 1e-6)))


def _reference_time(ta: Tracklet, tb: Tracklet) -> float:
    """
    Returns the best absolute time at which to compare the two tracks.
    Prefers the centre of overlap; falls back to the nearest boundary.
    """
    overlap_start = max(ta.start_time, tb.start_time)
    overlap_end   = min(ta.end_time,   tb.end_time)
    if overlap_end >= overlap_start:
        return (overlap_start + overlap_end) / 2.0

    # No overlap — use the closest endpoints
    if ta.end_time < tb.start_time:
        return (ta.end_time + tb.start_time) / 2.0
    return (tb.end_time + ta.start_time) / 2.0


def _geo_score(
    ta: Tracklet, tb: Tracklet,
    t_ref: float,
    max_speed: float,
    geo_sigma: float,
) -> Optional[float]:
    """
    Geographic proximity score at reference time.
    Returns None if positions are unavailable or imply impossible speed.
    """
    ga = ta.geo_at(t_ref)
    gb = tb.geo_at(t_ref)
    if ga is None or gb is None:
        return None

    dist = geo_distance(ga, gb)

    # Hard speed gate: check if the distance implies impossible travel
    dt = abs(ta.end_time - tb.start_time) if ta.end_time < tb.start_time \
         else abs(tb.end_time - ta.start_time) if tb.end_time < ta.start_time \
         else 0.0
    if dt > 0 and dist / dt > max_speed:
        return None

    return float(np.exp(-dist / max(geo_sigma, 1e-6)))


def _velocity_score(
    ta: Tracklet, tb: Tracklet, t_ref: float, vel_sigma: float
) -> Optional[float]:
    """Velocity similarity at reference time. Returns None if velocities unavailable."""
    va = ta.vel_at(t_ref)
    vb = tb.vel_at(t_ref)
    if va is None or vb is None:
        return None
    diff = np.linalg.norm(np.array(va) - np.array(vb))
    return float(np.exp(-diff / max(vel_sigma, 1e-6)))


def _appearance_score(
    ta: Tracklet, tb: Tracklet, siamese_scorer=None
) -> float:
    """Appearance similarity via Siamese embedding or histogram."""
    if siamese_scorer is not None and ta.mean_emb is not None and tb.mean_emb is not None:
        denom = np.linalg.norm(ta.mean_emb) * np.linalg.norm(tb.mean_emb)
        if denom > 0:
            return float(np.dot(ta.mean_emb, tb.mean_emb) / denom)
        return -1.0

    if ta.mean_hist is not None and tb.mean_hist is not None:
        score = cv2.compareHist(
            ta.mean_hist.astype(np.float32).reshape(-1, 1),
            tb.mean_hist.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_INTERSECT,
        )
        # Normalize to [0,1] by dividing by min of the two hist sums
        norm = min(ta.mean_hist.sum(), tb.mean_hist.sum())
        return float(score / norm) if norm > 0 else 0.0

    return 0.0


def score_pair(
    ta: Tracklet, tb: Tracklet,
    max_time_gap:  float,
    max_speed:     float,
    geo_sigma:     float,
    vel_sigma:     float,
    w_geo:         float,
    w_vel:         float,
    w_app:         float,
    siamese_scorer=None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (coarse_temporal_score, full_combined_score).
    Either may be None if hard gates are violated.
    """
    t_score = _temporal_score(ta, tb, max_time_gap)
    if t_score is None:
        return None, None   # outside time gate — skip entirely

    t_ref   = _reference_time(ta, tb)
    g_score = _geo_score(ta, tb, t_ref, max_speed, geo_sigma)
    v_score = _velocity_score(ta, tb, t_ref, vel_sigma)
    a_score = _appearance_score(ta, tb, siamese_scorer)

    # If geo is available and failed the speed gate, reject
    if g_score is None and (ta.records[0].geo_pos is not None or
                            tb.records[0].geo_pos is not None):
        return t_score, None

    # Normalise weights to those with data
    components, weights = [], []
    if g_score is not None:
        components.append(g_score); weights.append(w_geo)
    if v_score is not None:
        components.append(v_score); weights.append(w_vel)
    components.append(a_score); weights.append(w_app)

    total_w = sum(weights)
    combined = sum(c * w for c, w in zip(components, weights)) / total_w if total_w > 0 else 0.0
    return t_score, float(combined)


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def match_tracklets(
    all_tracklets: List[Tracklet],
    max_time_gap:  float,
    max_speed:     float,
    geo_sigma:     float,
    vel_sigma:     float,
    w_geo:         float,
    w_vel:         float,
    w_app:         float,
    match_threshold: float,
    siamese_scorer=None,
) -> Dict[int, int]:
    """
    Greedy offline cross-camera matching.

    Algorithm
    ---------
    1. Generate all cross-camera tracklet pairs.
    2. Compute cheap temporal score; filter by time gate; sort best-first.
    3. For each candidate pair (greedy):
       a. Check camera-collision safety via union-find.
       b. Compute full score.
       c. If score >= threshold: merge global IDs.

    Returns {tracklet_uid: global_id}.
    """
    # Assign unique integer UIDs to each tracklet
    uid_map: Dict[int, Tracklet] = {}   # uid -> tracklet
    uid = 0
    for t in all_tracklets:
        t._uid = uid     # type: ignore[attr-defined]
        uid_map[uid] = t
        uid += 1

    uf = UnionFind()
    for t in all_tracklets:
        uf.add(t._uid, t.cam)   # type: ignore[attr-defined]

    # Generate cross-camera pairs and compute coarse temporal scores
    n = len(all_tracklets)
    candidates = []   # (temporal_score, uid_a, uid_b)

    for i in range(n):
        for j in range(i + 1, n):
            ta, tb = all_tracklets[i], all_tracklets[j]
            if ta.cam == tb.cam:
                continue
            t_score = _temporal_score(ta, tb, max_time_gap)
            if t_score is not None:
                candidates.append((t_score, ta._uid, tb._uid))   # type: ignore[attr-defined]

    # Sort: highest temporal score first (most temporally compatible)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Track how many matches each (uid, cam) pair already has
    # Each tracklet may match at most once per OTHER camera
    matched_pairs: set = set()   # (uid_a, cam_b) already matched

    for t_score, uid_a, uid_b in candidates:
        ta, tb = uid_map[uid_a], uid_map[uid_b]

        # Skip if this (tracklet, camera) pair already has a match
        if (uid_a, tb.cam) in matched_pairs or (uid_b, ta.cam) in matched_pairs:
            continue

        if not uf.can_union(uid_a, uid_b):
            continue

        _, full_score = score_pair(
            ta, tb,
            max_time_gap=max_time_gap, max_speed=max_speed,
            geo_sigma=geo_sigma, vel_sigma=vel_sigma,
            w_geo=w_geo, w_vel=w_vel, w_app=w_app,
            siamese_scorer=siamese_scorer,
        )

        if full_score is not None and full_score >= match_threshold:
            uf.union(uid_a, uid_b)
            matched_pairs.add((uid_a, tb.cam))
            matched_pairs.add((uid_b, ta.cam))

    # Assign sequential global IDs to groups spanning ≥ 2 cameras
    groups = uf.groups()
    result: Dict[int, int] = {}
    next_gid = 1

    for root, members in groups.items():
        cams_in_group = {uid_map[m].cam for m in members}
        if len(cams_in_group) < 2:
            continue   # single-camera: discard
        for m in members:
            result[m] = next_gid
        next_gid += 1

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_output(
    out_dir: str,
    cam: str,
    tracklets: Dict[int, Tracklet],
    uid_to_gid: Dict[int, int],
):
    """Write per-camera MOTChallenge txt with global IDs for matched tracks."""
    cam_dir = os.path.join(out_dir, cam)
    os.makedirs(cam_dir, exist_ok=True)
    out_path = os.path.join(cam_dir, "offline_mtmc.txt")

    lines = []
    for local_id, t in tracklets.items():
        uid = t._uid   # type: ignore[attr-defined]
        if uid not in uid_to_gid:
            continue
        gid = uid_to_gid[uid]
        for rec in t.records:
            x, y, w, h = rec.bbox
            frame_1idx = rec.frame_0idx + 1
            lines.append(f"{frame_1idx},{gid},{int(x)},{int(y)},{int(w)},{int(h)},1,-1,-1,-1\n")

    lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    with open(out_path, "w") as f:
        f.writelines(lines)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_offline_mtmc(args):
    scenario_name = os.path.basename(args.scenario_dir.rstrip("/"))
    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "results", "offline", scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    cameras = discover_cameras(args.scenario_dir)
    if not cameras:
        raise RuntimeError(f"No cameras found in {args.scenario_dir}")

    print(f"\n{'='*55}")
    print(f"  Offline MTMC  |  {scenario_name}  |  {len(cameras)} cameras")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}\n")

    # --- Timestamps & homographies ---
    ts_path = os.path.normpath(
        os.path.join(args.scenario_dir, "..", "..", "cam_timestamp", f"{scenario_name}.txt")
    )
    timestamps = load_timestamps(ts_path)
    if not timestamps:
        print(f"  [WARNING] No timestamps found at {ts_path} — using offset=0 for all cameras")

    def cam_fps(cam: str) -> float:
        return 8.0 if cam == "c015" and scenario_name == "S03" else 10.0

    homographies = {}
    for cam in cameras:
        cal = os.path.join(args.scenario_dir, cam, "calibration.txt")
        homographies[cam] = load_homography(cal)

    # --- Siamese scorer (optional) ---
    siamese_scorer = None
    if args.scorer == "siamese" and args.siamese_ckpt:
        print(f"  Loading Siamese model from {args.siamese_ckpt} ...")
        siamese_scorer = SiameseScorer(args.siamese_ckpt, device=args.device)

    # --- Per-camera tracking ---
    all_tracklets: List[Tracklet] = []
    cam_tracklets: Dict[str, Dict[int, Tracklet]] = {}
    gt_tracks_per_cam: Dict[str, dict] = {}
    all_raw_results: Dict[str, List] = {cam: [] for cam in cameras}  # raw local tracks

    for cam in cameras:
        print(f"  [{cam}] Tracking ...")
        fps    = cam_fps(cam)
        offset = timestamps.get(cam, 0.0)

        det_path = find_detection_file(args.scenario_dir, cam, args.dets_dir)
        roi_mask = load_roi_mask(args.scenario_dir, cam) if args.use_roi else None
        cam_dets = load_detections(det_path, conf_thr=args.conf_thr, roi_mask=roi_mask)

        # Build frame_to_abs mapping
        frame_to_abs = {f: offset + f / fps for f in cam_dets}

        # Run tracker (no ReID — pure local tracking)
        if args.tracker == "bytetrack":
            tracker = ByteTrackMTMCTracker(
                track_thresh=args.bt_track_thresh,
                track_buffer=args.bt_track_buffer,
                match_thresh=args.bt_match_thresh,
                fps=fps, timestamp_offset=offset,
            )
        else:
            tracker = SORTTracker(
                max_age=args.max_age, min_hits=args.min_hits,
                iou_threshold=args.iou_thr,
                fps=fps, timestamp_offset=offset,
            )

        vid_path = os.path.join(args.scenario_dir, cam, "vdo.avi")
        cap = cv2.VideoCapture(vid_path) if os.path.exists(vid_path) else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else max(cam_dets, default=0) + 1

        tracker_output: Dict[int, List[Tuple]] = {}
        frames_cache:   Dict[int, Optional[np.ndarray]] = {}

        for frame_0idx in tqdm(range(total_frames), desc=f"    {cam}", leave=False):
            frame = None
            if cap:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            dets = cam_dets.get(frame_0idx, [])
            tracked = tracker.track(dets, frame_id=frame_0idx, frame=frame)
            tracker_output[frame_0idx] = tracked

            # Accumulate raw results for debug video (1-indexed frame, local IDs)
            for local_id, bbox in tracked:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                all_raw_results[cam].append((frame_0idx + 1, local_id, x, y, w, h))

            # Cache every Nth frame for appearance (avoid keeping all in memory)
            if frame is not None and frame_0idx % args.sample_every == 0:
                frames_cache[frame_0idx] = frame

        if cap:
            cap.release()

        # Build tracklets
        tracklets = build_tracklets(
            tracker_output=tracker_output,
            frame_to_abs=frame_to_abs,
            homography=homographies.get(cam),
            frames=frames_cache,
            cam=cam,
            velocity_window=args.velocity_window,
        )

        # Filter parked
        before = len(tracklets)
        tracklets = filter_parked(tracklets, min_displacement_px=args.min_displacement)
        print(f"  [{cam}] {before} tracks → {len(tracklets)} after parked filter")

        # Build appearance metadata
        for t in tracklets.values():
            build_tracklet_meta(t, histogram_scorer, siamese_scorer,
                                sample_every=args.sample_every)

        cam_tracklets[cam] = tracklets
        all_tracklets.extend(tracklets.values())

        # Load GT for evaluation
        gt_path = os.path.join(args.scenario_dir, cam, "gt", "gt.txt")
        if os.path.exists(gt_path) and load_gt_tracks:
            gt_tracks_per_cam[cam] = load_gt_tracks(gt_path, train_frames=0)

    # --- Raw debug video (local tracks, no pruning) ---
    if getattr(args, "raw_video", False) or getattr(args, "raw_montage", False):
        all_local_ids = {cam: set(lid for _, lid, *_ in all_raw_results[cam])
                         for cam in cameras}
        valid_raw = set().union(*all_local_ids.values())
        # Temporarily redirect flags so render_outputs uses raw_ variants
        import copy as _copy
        raw_args = _copy.copy(args)
        raw_args.write_video = getattr(args, "raw_video", False)
        raw_args.montage     = getattr(args, "raw_montage", False)
        raw_out = os.path.join(out_dir, "raw")
        os.makedirs(raw_out, exist_ok=True)
        raw_args.out_dir = raw_out
        print(f"\n  [raw video] Rendering local tracks → {raw_out}")
        render_outputs(raw_args, cameras, all_raw_results, valid_raw,
                       timestamps, cam_fps, raw_out)

    # --- Cross-camera matching ---
    print(f"\n  Matching {len(all_tracklets)} tracklets across {len(cameras)} cameras ...")
    uid_to_gid = match_tracklets(
        all_tracklets=all_tracklets,
        max_time_gap=args.max_time_gap,
        max_speed=args.max_speed,
        geo_sigma=args.geo_sigma,
        vel_sigma=args.vel_sigma,
        w_geo=args.w_geo,
        w_vel=args.w_vel,
        w_app=args.w_app,
        match_threshold=args.match_thr,
        siamese_scorer=siamese_scorer,
    )

    multi_cam_ids = set(uid_to_gid.values())
    print(f"  {len(multi_cam_ids)} multi-camera global IDs found")

    # --- Write output & evaluate ---
    metrics_rows = {}
    for cam in cameras:
        tracklets = cam_tracklets.get(cam, {})
        out_path  = write_output(out_dir, cam, tracklets, uid_to_gid)

        # Build pred_tracks for evaluation: {frame_1idx: [(gid, bbox)]}
        pred_tracks = defaultdict(list)
        for t in tracklets.values():
            uid = t._uid   # type: ignore[attr-defined]
            if uid not in uid_to_gid:
                continue
            gid = uid_to_gid[uid]
            for rec in t.records:
                pred_tracks[rec.frame_0idx + 1].append(
                    (gid, tuple(int(v) for v in rec.bbox))
                )

        if args.eval and cam in gt_tracks_per_cam and evaluate_tracking:
            try:
                m = evaluate_tracking(dict(pred_tracks), gt_tracks_per_cam[cam])
                metrics_rows[cam] = m
                print(f"  [{cam}]  HOTA={m['HOTA']:.4f}  IDF1={m['IDF1']:.4f}")
            except Exception as e:
                print(f"  [{cam}]  Eval error: {e}")

        print(f"  [{cam}] → {out_path}")

    if metrics_rows:
        mean_hota = float(np.mean([m["HOTA"] for m in metrics_rows.values()]))
        mean_idf1 = float(np.mean([m["IDF1"] for m in metrics_rows.values()]))
        print(f"\n  {'─'*45}")
        print(f"  Mean  HOTA={mean_hota:.4f}  IDF1={mean_idf1:.4f}")

        if args.metrics_csv:
            import csv
            os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
            with open(args.metrics_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["camera", "HOTA", "IDF1"])
                w.writeheader()
                for cam, m in metrics_rows.items():
                    w.writerow({"camera": cam, **m})
                w.writerow({"camera": "mean", "HOTA": mean_hota, "IDF1": mean_idf1})
            print(f"  Metrics → {args.metrics_csv}")

        print()
        ret = {"HOTA": mean_hota, "IDF1": mean_idf1}
    else:
        ret = None

    # ------------------------------------------------------------------
    # Video rendering (per-camera MP4 and/or montage)
    # ------------------------------------------------------------------
    if getattr(args, "write_video", False) or getattr(args, "montage", False):
        # Build all_results in the format render_outputs expects:
        # {cam: [(frame_1idx, gid, x, y, w, h), ...]}
        valid_ids = set(uid_to_gid.values())
        all_results = {cam: [] for cam in cameras}
        for cam in cameras:
            for t in cam_tracklets.get(cam, {}).values():
                uid = t._uid  # type: ignore[attr-defined]
                if uid not in uid_to_gid:
                    continue
                gid = uid_to_gid[uid]
                for rec in t.records:
                    x, y, w, h = (int(rec.bbox[0]), int(rec.bbox[1]),
                                  int(rec.bbox[2]), int(rec.bbox[3]))
                    all_results[cam].append((rec.frame_0idx + 1, gid, x, y, w, h))

        render_outputs(args, cameras, all_results, valid_ids,
                       timestamps, cam_fps, out_dir)

    # ------------------------------------------------------------------
    # AIC official evaluation (IDF1 via motmetrics, all cameras jointly)
    # ------------------------------------------------------------------
    if getattr(args, "aic_eval", False):
        if not _HAVE_AIC:
            print("  [AIC eval] Skipped – aic_eval module not importable")
        else:
            aic_gt_path = getattr(args, "aic_gt", None) or os.path.normpath(
                os.path.join(SCRIPT_DIR, "..", "AI_CITY_CHALLENGE_2022_TRAIN",
                             "eval", "ground_truth_train.txt")
            )
            gt_df = load_aic_gt_for_sequence(aic_gt_path, scenario_name)
            pred_df = merge_cam_outputs(out_dir, cameras, filename="offline_mtmc.txt")
            aic_res = run_aic_eval(pred_df, gt_df)
            print_aic_results(aic_res, label=f"AIC  {scenario_name}")
            if ret is not None:
                ret["AIC_IDF1"] = aic_res["idf1"]

    return ret


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(args):
    """
    Sweep over all combinations of --gs_* parameter lists, run run_offline_mtmc
    for each, and report the best configuration by mean HOTA.

    Parameters not supplied via --gs_* keep their fixed values from the
    corresponding base flags (e.g. --match_thr fixes that value).

    Output
    ------
    A CSV (--gs_csv or <out_dir>/gs_results.csv) with one row per combination,
    plus a summary of the best config printed to stdout.
    """
    import copy
    import itertools
    import contextlib
    import csv as _csv

    scenario_name = os.path.basename(args.scenario_dir.rstrip("/\\"))
    base_out = args.out_dir or os.path.join(SCRIPT_DIR, "results", "offline", scenario_name)
    gs_csv_path = args.gs_csv or os.path.join(base_out, "gs_results.csv")
    os.makedirs(base_out, exist_ok=True)

    # (base_param, gs_attr, type)
    PARAM_FIELDS = [
        # Detection filtering
        ("conf_thr",         "gs_conf_thr",         float),
        # Tracker
        ("iou_thr",          "gs_iou_thr",          float),
        ("max_age",          "gs_max_age",           int),
        ("min_hits",         "gs_min_hits",          int),
        ("bt_track_thresh",  "gs_bt_track_thresh",   float),
        ("bt_track_buffer",  "gs_bt_track_buffer",   int),
        ("bt_match_thresh",  "gs_bt_match_thresh",   float),
        # Tracklet building
        ("velocity_window",  "gs_velocity_window",   int),
        ("min_displacement", "gs_min_displacement",  float),
        # Matching
        ("match_thr",        "gs_match_thr",         float),
        ("max_time_gap",     "gs_max_time_gap",      float),
        ("max_speed",        "gs_max_speed",         float),
        ("geo_sigma",        "gs_geo_sigma",         float),
        ("vel_sigma",        "gs_vel_sigma",         float),
        ("w_geo",            "gs_w_geo",             float),
        ("w_vel",            "gs_w_vel",             float),
        ("w_app",            "gs_w_app",             float),
    ]

    grid = {}
    for param, gs_attr, _ in PARAM_FIELDS:
        values = getattr(args, gs_attr, None)
        grid[param] = values if values else [getattr(args, param)]

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n      = len(combos)

    print(f"\n{'='*60}")
    print(f"  Offline grid search: {n} combinations  |  {scenario_name}")
    for k in keys:
        if len(grid[k]) > 1:
            print(f"    {k}: {grid[k]}")
    print(f"  Results → {gs_csv_path}")
    print(f"{'='*60}\n")

    csv_fields = keys + ["HOTA", "IDF1"]
    best_row = None

    with open(gs_csv_path, "w", newline="") as fcsv:
        writer = _csv.DictWriter(fcsv, fieldnames=csv_fields)
        writer.writeheader()

        for i, combo in enumerate(combos):
            run_args = copy.deepcopy(args)
            desc_parts = []
            for param, value in zip(keys, combo):
                setattr(run_args, param, value)
                if len(grid[param]) > 1:
                    desc_parts.append(f"{param}={value}")

            combo_tag = "_".join(desc_parts) or "default"
            run_args.out_dir     = os.path.join(base_out, "gs", combo_tag)
            run_args.eval        = True
            run_args.metrics_csv = None
            run_args.grid_search = False
            os.makedirs(run_args.out_dir, exist_ok=True)

            print(f"  [{i+1}/{n}] {combo_tag}", end=" ... ", flush=True)

            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stdout(devnull):
                result = run_offline_mtmc(run_args)

            if result is None:
                print("no GT – skipped")
                continue

            hota, idf1 = result["HOTA"], result["IDF1"]
            print(f"HOTA={hota:.4f}  IDF1={idf1:.4f}")

            row = dict(zip(keys, combo))
            row["HOTA"] = hota
            row["IDF1"]  = idf1
            writer.writerow(row)
            fcsv.flush()

            if best_row is None or hota > best_row["HOTA"]:
                best_row = row

    print(f"\n{'='*60}")
    if best_row:
        print(f"  Best config (HOTA={best_row['HOTA']:.4f}  IDF1={best_row['IDF1']:.4f}):")
        for k in keys:
            if len(grid[k]) > 1:
                print(f"    --{k} {best_row[k]}")
    print(f"  Full results → {gs_csv_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline MTMC tracker")

    p.add_argument("--scenario_dir", required=True)
    p.add_argument("--dets_dir",     default=None)
    p.add_argument("--out_dir",      default=None)
    p.add_argument("--eval",         action="store_true")
    p.add_argument("--metrics_csv",  default=None)

    # Detection filtering
    p.add_argument("--conf_thr",     default=0.0,  type=float)
    p.add_argument("--use_roi",      action="store_true")

    # Tracker
    p.add_argument("--tracker",      default="bytetrack", choices=["sort", "bytetrack"])
    p.add_argument("--iou_thr",      default=0.1458, type=float)
    p.add_argument("--max_age",      default=5,      type=int)
    p.add_argument("--min_hits",     default=3,      type=int)
    p.add_argument("--bt_track_thresh", default=0.25, type=float)
    p.add_argument("--bt_track_buffer", default=30,   type=int)
    p.add_argument("--bt_match_thresh", default=0.8,  type=float)

    # Tracklet building
    p.add_argument("--velocity_window",  default=5,    type=int,
                   help="Frames for moving-average velocity")
    p.add_argument("--sample_every",     default=5,    type=int,
                   help="Keep every Nth frame for appearance")
    p.add_argument("--min_displacement", default=50.0, type=float,
                   help="Min pixel displacement to not be considered parked")

    # Matching
    p.add_argument("--match_thr",    default=0.4,   type=float,
                   help="Minimum full score to accept a cross-camera match")
    p.add_argument("--max_time_gap", default=60.0,  type=float,
                   help="Max seconds between tracks to consider matching")
    p.add_argument("--max_speed",    default=30.0,  type=float,
                   help="Hard speed gate in m/s (~108 km/h)")
    p.add_argument("--geo_sigma",    default=50.0,  type=float,
                   help="Geo distance decay constant in metres")
    p.add_argument("--vel_sigma",    default=5.0,   type=float,
                   help="Velocity difference decay constant in m/s")
    p.add_argument("--w_geo",        default=0.4,   type=float)
    p.add_argument("--w_vel",        default=0.2,   type=float)
    p.add_argument("--w_app",        default=0.4,   type=float)

    # Appearance
    p.add_argument("--scorer",       default="histogram", choices=["histogram", "siamese"])
    p.add_argument("--siamese_ckpt", default=None)
    p.add_argument("--device",       default="cuda")

    # Video output (MTMC-filtered tracks)
    p.add_argument("--write_video", action="store_true",
                   help="Write per-camera annotated output videos (multi-cam tracks only)")
    p.add_argument("--montage",     action="store_true",
                   help="Write time-synchronised multi-camera montage video (multi-cam tracks only)")
    p.add_argument("--tile_w",      default=640, type=int,
                   help="Width of each camera tile in the montage (default 640)")
    p.add_argument("--tile_h",      default=360, type=int,
                   help="Height of each camera tile in the montage (default 360)")
    # Raw debug video (unfiltered local ByteTrack/SORT tracks)
    p.add_argument("--raw_video",   action="store_true",
                   help="Write per-camera debug videos with raw local tracks (no pruning)")
    p.add_argument("--raw_montage", action="store_true",
                   help="Write debug montage with raw local tracks (no pruning)")

    # AIC official evaluation
    p.add_argument("--aic_eval", action="store_true",
                   help="Run official AIC IDF1 evaluation (motmetrics) after tracking")
    p.add_argument("--aic_gt", default=None,
                   help="Path to AIC ground_truth_train.txt "
                        "(default: ../AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt)")

    # Grid search
    p.add_argument("--grid_search",  action="store_true",
                   help="Sweep --gs_* param grids and report best config by HOTA")
    p.add_argument("--gs_csv",       default=None,
                   help="Save grid search results to this CSV")

    # Each --gs_<param> takes one or more values; omitting fixes that param at its default
    p.add_argument("--gs_conf_thr",         nargs="+", type=float, default=None)
    p.add_argument("--gs_iou_thr",          nargs="+", type=float, default=None)
    p.add_argument("--gs_max_age",          nargs="+", type=int,   default=None)
    p.add_argument("--gs_min_hits",         nargs="+", type=int,   default=None)
    p.add_argument("--gs_bt_track_thresh",  nargs="+", type=float, default=None)
    p.add_argument("--gs_bt_track_buffer",  nargs="+", type=int,   default=None)
    p.add_argument("--gs_bt_match_thresh",  nargs="+", type=float, default=None)
    p.add_argument("--gs_velocity_window",  nargs="+", type=int,   default=None)
    p.add_argument("--gs_min_displacement", nargs="+", type=float, default=None)
    p.add_argument("--gs_match_thr",        nargs="+", type=float, default=None)
    p.add_argument("--gs_max_time_gap",     nargs="+", type=float, default=None)
    p.add_argument("--gs_max_speed",        nargs="+", type=float, default=None)
    p.add_argument("--gs_geo_sigma",        nargs="+", type=float, default=None)
    p.add_argument("--gs_vel_sigma",        nargs="+", type=float, default=None)
    p.add_argument("--gs_w_geo",            nargs="+", type=float, default=None)
    p.add_argument("--gs_w_vel",            nargs="+", type=float, default=None)
    p.add_argument("--gs_w_app",            nargs="+", type=float, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.grid_search:
        grid_search(args)
    else:
        run_offline_mtmc(args)
