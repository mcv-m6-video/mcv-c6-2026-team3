"""
main_mtmc.py
------------
Multi-Target Multi-Camera (MTMC) tracking entry point.

Runs SORT + Cross-Camera ReID across all cameras of a given scenario
(e.g. S01 which contains c001–c005) in a frame-synchronised loop.

Key differences from the single-camera main.py
===============================================
* One `GlobalIDManager` is shared by all cameras → IDs are unique globally.
* One `CrossCameraReIDMatcher` is shared → new detections are matched against
  recent appearances from every other camera before a fresh ID is issued.
* Output is written to per-camera .txt files using global IDs, so the result
  is directly compatible with the CityFlow MTMC evaluation format.

Usage
-----
    python main_mtmc.py --scenario_dir  /data/train/S01 \
                        --dets_dir      results/S01 \
                        --out_dir       results/S01 \
                        --match_thr     0.60 \
                        --lookback      30

Scorer swap example
-------------------
    from cross_camera_reid import CrossCameraReIDMatcher, GlobalIDManager
    from my_deep_reid import deep_scorer          # your custom scorer

    gid  = GlobalIDManager()
    reid = CrossCameraReIDMatcher(gid,
                                  scorer_fn=deep_scorer,
                                  match_threshold=0.75,
                                  lookback_frames=60)
    # then build SORTTrackers as shown in _build_trackers() below
"""

import argparse
import os
import sys
import glob

import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Local modules
from cross_camera_reid import (
    CrossCameraReIDMatcher,
    GlobalIDManager,
    histogram_scorer,
)
from sort_tracker import SORTTracker, ByteTrackMTMCTracker
from geo_utils import load_homography, load_timestamps

try:
    from evaluation.tracking_eval import load_gt_tracks, evaluate_tracking
except ImportError as e:
    print(f"[WARNING] Evaluation module not available: {e}")
    load_gt_tracks = None
    evaluate_tracking = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MTMC SORT + ReID tracker")

    p.add_argument("--scenario_dir", required=True,
                   help="Path to scenario directory, e.g. /data/train/S01")
    p.add_argument("--dets_dir", default=None,
                   help="Root dir for detection .txt files. "
                        "Expects <dets_dir>/<cam>/detections.txt. "
                        "Defaults to <scenario_dir>/<cam>/det/det_yolo3.txt")
    p.add_argument("--out_dir", default=None,
                   help="Output directory. Defaults to ./results/<scenario>")
    p.add_argument("--write_video", action="store_true",
                   help="Write per-camera annotated output videos")
    p.add_argument("--montage", action="store_true",
                   help="Write time-synchronised multi-camera montage video")
    p.add_argument("--tile_w", default=640, type=int,
                   help="Width of each camera tile in the montage (default 640)")
    p.add_argument("--tile_h", default=360, type=int,
                   help="Height of each camera tile in the montage (default 360)")

    # Tracker selection
    p.add_argument("--tracker", default="sort", choices=["sort", "bytetrack"],
                   help="Single-camera tracker backend (default: sort)")

    # SORT hyper-parameters
    p.add_argument("--iou_thr",   default=0.1458, type=float)
    p.add_argument("--max_age",   default=5,       type=int)
    p.add_argument("--min_hits",  default=12,      type=int)

    # ByteTrack hyper-parameters (used when --tracker bytetrack)
    p.add_argument("--bt_track_thresh", default=0.25, type=float,
                   help="ByteTrack high-confidence detection threshold")
    p.add_argument("--bt_track_buffer", default=30,   type=int,
                   help="ByteTrack frames to keep a lost track alive")
    p.add_argument("--bt_match_thresh", default=0.8,  type=float,
                   help="ByteTrack internal IoU matching threshold")

    # ReID hyper-parameters
    p.add_argument("--match_thr", default=0.30,  type=float,
                   help="Minimum combined score to accept a cross-camera match")
    p.add_argument("--lookback",  default=3.0,   type=float,
                   help="Seconds of history to search in other cameras")

    # Evaluation
    p.add_argument("--eval", action="store_true",
                   help="Evaluate pruned global tracks against per-camera GT")
    p.add_argument("--metrics_csv", default=None,
                   help="Save per-camera + mean evaluation metrics to this CSV file")

    # Geographic enhancement
    p.add_argument("--geo_weight", default=0.3, type=float,
                   help="Blend weight for geo vs appearance [0=appearance only, 1=geo only]")
    p.add_argument("--max_speed", default=30.0, type=float,
                   help="Max plausible speed in metres/sec (hard gate). Default 30 m/s ≈ 108 km/h.")
    p.add_argument("--geo_sigma", default=50.0, type=float,
                   help="Geo distance-decay constant in metres: score = exp(-dist/geo_sigma).")
    p.add_argument("--velocity_window", default=5, type=int,
                   help="GPS samples for per-track moving-average velocity (default 5)")

    # ------------------------------------------------------------------
    # Grid search  (--grid_search enables; --gs_<param> specifies values)
    # ------------------------------------------------------------------
    p.add_argument("--grid_search", action="store_true",
                   help="Sweep over --gs_* param grids and report best config")
    p.add_argument("--gs_csv", default=None,
                   help="Save full grid search results to this CSV (default: gs_results.csv in out_dir)")

    # Each --gs_* takes one or more values; omitting it fixes that param at its default
    p.add_argument("--gs_match_thr",       nargs="+", type=float, default=None)
    p.add_argument("--gs_lookback",        nargs="+", type=float, default=None)
    p.add_argument("--gs_geo_weight",      nargs="+", type=float, default=None)
    p.add_argument("--gs_max_speed",       nargs="+", type=float, default=None)
    p.add_argument("--gs_geo_sigma",       nargs="+", type=float, default=None)
    p.add_argument("--gs_velocity_window", nargs="+", type=int,   default=None)
    p.add_argument("--gs_iou_thr",         nargs="+", type=float, default=None)
    p.add_argument("--gs_max_age",         nargs="+", type=int,   default=None)
    p.add_argument("--gs_min_hits",        nargs="+", type=int,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_cameras(scenario_dir: str):
    """Return sorted list of camera subdirectory names (e.g. ['c001','c002'])."""
    cams = sorted([
        d for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith("c")
    ])
    return cams


def load_detections(path: str):
    """
    Load detections from a MOTChallenge-format .txt file.
    Returns {frame_id (0-indexed): [[x, y, w, h], ...]}
    """
    dets = defaultdict(list)
    if not os.path.exists(path):
        print(f"  [WARNING] Detection file not found: {path}")
        return dets
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # MOT format: frame, id_or_-1, left, top, width, height, conf, ...
            frame_1idx = int(float(parts[0]))
            x, y, w, h = (float(parts[i]) for i in (2, 3, 4, 5))
            dets[frame_1idx - 1].append([x, y, w, h])   # convert to 0-indexed
    return dets


def find_detection_file(scenario_dir: str, cam: str, dets_dir=None) -> str:
    """
    Resolve detection file path with a sensible fallback chain:
        1. <dets_dir>/<cam>/detections.txt          (user override)
        2. <scenario_dir>/<cam>/det/det_yolo3.txt   (dataset default)
        3. first *.txt inside <scenario_dir>/<cam>/det/
    """
    if dets_dir:
        p = os.path.join(dets_dir, cam, "detections.txt")
        if os.path.exists(p):
            return p

    base = os.path.join(scenario_dir, cam, "det")
    for name in ("det_yolo3.txt", "det_mask_rcnn.txt", "det_ssd512.txt"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p

    candidates = glob.glob(os.path.join(base, "*.txt"))
    if candidates:
        return sorted(candidates)[0]

    return os.path.join(base, "detections.txt")   # will trigger WARNING


# ---------------------------------------------------------------------------
# Video rendering helpers
# ---------------------------------------------------------------------------

def _id_color(tid: int):
    """Deterministic BGR color for a global track ID (HSV color wheel)."""
    hue = (tid * 47) % 180
    hsv = np.uint8([[[hue, 220, 220]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _draw_tracks(frame, frame_results, scale_x: float = 1.0, scale_y: float = 1.0):
    """Draw color-coded global-ID boxes onto frame (in-place)."""
    for tid, x, y, w, h in frame_results:
        color = _id_color(tid)
        x1 = int(x * scale_x);  y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x);  y2 = int((y + h) * scale_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(tid), (x1, max(y1 - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def render_outputs(args, cameras, all_results, valid_ids,
                   timestamps, cam_fps_fn, out_dir):
    """
    Second pass over the videos to render per-camera MP4s and/or a
    time-synchronised montage.

    Montage layout
    --------------
    * Auto grid: ceil(sqrt(N)) columns.
    * Each tile is ``args.tile_w × args.tile_h`` pixels (default 640×360).
    * Cameras not yet started / already ended show a black tile.
    * Camera label (white) drawn top-left of each tile.
    * Bounding boxes color-coded by global ID.
    * FPS fixed at 10 for the montage.
    """
    TILE_W   = args.tile_w
    TILE_H   = args.tile_h
    MON_FPS  = 10.0

    # Per-camera lookup: {cam: {frame_1idx: [(tid, x, y, w, h)]}}
    cam_fr = {}
    for cam in cameras:
        cam_fr[cam] = defaultdict(list)
        for (frame_id, tid, x, y, w, h) in all_results[cam]:
            if tid in valid_ids:
                cam_fr[cam][frame_id].append((tid, x, y, w, h))

    # Open video captures and collect frame counts
    caps_r       = {}
    frame_counts = {}
    vid_sizes    = {}
    for cam in cameras:
        vp = os.path.join(args.scenario_dir, cam, "vdo.avi")
        if os.path.exists(vp):
            cap = cv2.VideoCapture(vp)
            caps_r[cam]       = cap
            frame_counts[cam] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_sizes[cam]    = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if not caps_r:
        print("  [render] No video files found – skipping render.")
        return

    offsets = {cam: timestamps.get(cam, 0.0) for cam in cameras}

    # Global time range: from t=0 (earliest camera) to last frame of last camera
    t_end = max(
        offsets[cam] + frame_counts[cam] / cam_fps_fn(cam)
        for cam in cameras if cam in caps_r
    )
    n_montage = int(np.ceil(t_end * MON_FPS))

    # Per-camera video writers (optional)
    cam_writers = {}
    if args.write_video:
        for cam in cameras:
            if cam not in caps_r:
                continue
            mp4_path = os.path.join(out_dir, cam, "tracking.mp4")
            cam_writers[cam] = cv2.VideoWriter(
                mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                cam_fps_fn(cam), (TILE_W, TILE_H)
            )

    # Montage writer (optional)
    montage_writer = None
    if args.montage:
        n_cams = len(cameras)
        n_cols = int(np.ceil(np.sqrt(n_cams)))
        n_rows = int(np.ceil(n_cams / n_cols))
        mon_path = os.path.join(out_dir, "montage.mp4")
        montage_writer = cv2.VideoWriter(
            mon_path, cv2.VideoWriter_fourcc(*"mp4v"),
            MON_FPS, (n_cols * TILE_W, n_rows * TILE_H)
        )
        print(f"\n  [render] Montage: {n_cols}×{n_rows} grid → {mon_path}")

    # Per-camera read state: cap position (0-indexed next-to-read)
    cap_pos      = {cam: 0 for cam in caps_r}
    current_frm  = {cam: None for cam in caps_r}  # last successfully read frame

    def get_tile(cam: str, montage_t: float):
        """Return a TILE_W×TILE_H BGR tile for camera `cam` at scenario time `t`."""
        offset  = offsets[cam]
        fps     = cam_fps_fn(cam)
        cam_t   = montage_t - offset

        if cam not in caps_r or cam_t < 0:
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

        target = int(cam_t * fps)   # 0-indexed
        if target >= frame_counts.get(cam, 0):
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

        # Advance cap sequentially (target is non-decreasing across montage frames)
        cap = caps_r[cam]
        while cap_pos[cam] <= target:
            ret, frm = cap.read()
            if not ret:
                current_frm[cam] = None
                break
            current_frm[cam] = frm
            cap_pos[cam] += 1

        frm = current_frm[cam]
        if frm is None:
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

        orig_w, orig_h = vid_sizes[cam]
        tile = cv2.resize(frm, (TILE_W, TILE_H))

        frame_1idx = target + 1
        _draw_tracks(tile, cam_fr[cam].get(frame_1idx, []),
                     scale_x=TILE_W / orig_w, scale_y=TILE_H / orig_h)

        # Camera label
        cv2.putText(tile, cam, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return tile

    # ---- Render loop ----
    print(f"  [render] Rendering {n_montage} montage frames "
          f"({t_end:.1f}s at {MON_FPS:.0f} fps)...")

    # For per-camera write_video we do a separate sequential pass per camera
    # (avoids re-reading frames needed by the montage at different offsets).
    # The montage loop handles seeking correctly via cap_pos tracking.

    for m in tqdm(range(n_montage), desc="Render"):
        t = m / MON_FPS

        if args.montage:
            tiles = []
            for cam in cameras:
                tiles.append(get_tile(cam, t))

            # Pad to full grid with black tiles
            while len(tiles) < n_cols * n_rows:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

            rows = []
            for r in range(n_rows):
                rows.append(np.hstack(tiles[r * n_cols:(r + 1) * n_cols]))
            montage_writer.write(np.vstack(rows))

    # Per-camera videos: re-read independently so seeking is sequential
    if args.write_video and cam_writers:
        print("  [render] Writing per-camera annotated videos...")
        for cam, writer in cam_writers.items():
            vp  = os.path.join(args.scenario_dir, cam, "vdo.avi")
            cap = cv2.VideoCapture(vp)
            orig_w, orig_h = vid_sizes[cam]
            f_idx = 0
            while True:
                ret, frm = cap.read()
                if not ret:
                    break
                tile = cv2.resize(frm, (TILE_W, TILE_H))
                frame_1idx = f_idx + 1
                _draw_tracks(tile, cam_fr[cam].get(frame_1idx, []),
                             scale_x=TILE_W / orig_w, scale_y=TILE_H / orig_h)
                cv2.putText(tile, cam, (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(tile)
                f_idx += 1
            cap.release()

    # Cleanup
    for cap in caps_r.values():
        cap.release()
    for w in cam_writers.values():
        w.release()
    if montage_writer:
        montage_writer.release()
        print(f"  [render] Montage saved → {mon_path}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_mtmc(args):
    scenario_name = os.path.basename(args.scenario_dir.rstrip("/"))
    out_dir = args.out_dir or os.path.join(SCRIPT_DIR, "results", scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    cameras = discover_cameras(args.scenario_dir)
    if not cameras:
        raise RuntimeError(f"No camera subdirectories found in {args.scenario_dir}")

    print(f"\n{'='*50}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Cameras  : {cameras}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # 1.  Load per-camera calibrations, timestamps, and FPS
    # ------------------------------------------------------------------
    homographies = {}
    for cam in cameras:
        cal_path = os.path.join(args.scenario_dir, cam, "calibration.txt")
        H = load_homography(cal_path)
        homographies[cam] = H
        status = "loaded" if H is not None else "NOT FOUND"
        print(f"  [{cam}] Calibration: {status}")

    # Camera start-time offsets (seconds).  c015 in S03 is 8 fps; all others 10.
    ts_path = os.path.join(
        args.scenario_dir, "..", "..", "cam_timestamp", f"{scenario_name}.txt"
    )
    ts_path = os.path.normpath(ts_path)
    timestamps = load_timestamps(ts_path)
    if timestamps:
        print(f"\n  Timestamps loaded from {ts_path}")
        for cam in cameras:
            print(f"    [{cam}] offset = {timestamps.get(cam, 0.0):.3f} s")
    else:
        print(f"\n  [WARNING] Timestamp file not found at {ts_path} – using offset=0 for all cameras")

    def cam_fps(cam: str) -> float:
        """Return the frame rate for a camera (c015 in S03 is 8 fps)."""
        if cam == "c015" and scenario_name == "S03":
            return 8.0
        return 10.0

    # Detect per-camera FPS from video header as a cross-check
    for cam in cameras:
        vid_path = os.path.join(args.scenario_dir, cam, "vdo.avi")
        if os.path.exists(vid_path):
            _cap = cv2.VideoCapture(vid_path)
            _fps = _cap.get(cv2.CAP_PROP_FPS)
            _cap.release()
            if _fps and abs(_fps - cam_fps(cam)) > 0.5:
                print(f"  [{cam}] WARNING: video reports {_fps:.1f} fps but using {cam_fps(cam):.1f}")

    geo_active = any(H is not None for H in homographies.values())
    print(f"\n  Geo scoring  : {'ENABLED' if geo_active else 'DISABLED (no calibration found)'}")
    if geo_active:
        print(f"  geo_weight={args.geo_weight}  max_speed={args.max_speed} m/s  geo_sigma={args.geo_sigma} m\n")

    # ------------------------------------------------------------------
    # 2.  Shared ReID components
    # ------------------------------------------------------------------
    global_id_mgr = GlobalIDManager(start=1)

    reid_matcher = CrossCameraReIDMatcher(
        global_id_mgr=global_id_mgr,
        scorer_fn=histogram_scorer,         # ← swap this to use a different scorer
        match_threshold=args.match_thr,
        lookback_seconds=args.lookback,
        max_speed=args.max_speed,
        geo_sigma=args.geo_sigma,
        geo_weight=args.geo_weight if geo_active else 0.0,
        velocity_window=args.velocity_window,
    )

    # ------------------------------------------------------------------
    # 3.  Per-camera resources
    # ------------------------------------------------------------------
    trackers   = {}
    detections = {}
    caps       = {}
    # all_results[cam] = [(frame_1idx, global_id, x, y, w, h), ...]
    all_results   = defaultdict(list)
    # id_cameras[global_id] = {cam, cam, ...}  – for single-camera pruning
    id_cameras    = defaultdict(set)

    for cam in cameras:
        cam_out = os.path.join(out_dir, cam)
        os.makedirs(cam_out, exist_ok=True)

        # Tracker – each camera gets its own instance wired to the shared ReID
        if args.tracker == "bytetrack":
            trackers[cam] = ByteTrackMTMCTracker(
                track_thresh=args.bt_track_thresh,
                track_buffer=args.bt_track_buffer,
                match_thresh=args.bt_match_thresh,
                fps=cam_fps(cam),
                timestamp_offset=timestamps.get(cam, 0.0),
                reid_matcher=reid_matcher,
                camera_id=cam,
                global_id_mgr=global_id_mgr,
                homography=homographies.get(cam),
            )
        else:
            trackers[cam] = SORTTracker(
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_thr,
                reid_matcher=reid_matcher,
                camera_id=cam,
                global_id_mgr=global_id_mgr,
                homography=homographies.get(cam),
                fps=cam_fps(cam),
                timestamp_offset=timestamps.get(cam, 0.0),
            )

        # Detections
        det_path = find_detection_file(args.scenario_dir, cam, args.dets_dir)
        print(f"  [{cam}] Using detections: {det_path}")
        detections[cam] = load_detections(det_path)

        # Video capture
        vid_path = os.path.join(args.scenario_dir, cam, "vdo.avi")
        if os.path.exists(vid_path):
            caps[cam] = cv2.VideoCapture(vid_path)
        else:
            caps[cam] = None
            print(f"  [{cam}] WARNING: video not found at {vid_path}")

        pass  # txt files written after pruning

    # ------------------------------------------------------------------
    # 4.  Frame loop
    # ------------------------------------------------------------------
    # Determine total frames from the first available video
    total_frames = None
    for cam in cameras:
        if caps[cam] is not None:
            total_frames = int(caps[cam].get(cv2.CAP_PROP_FRAME_COUNT))
            break

    print(f"\nProcessing {total_frames or '?'} frames across {len(cameras)} cameras...\n")

    for f_id in tqdm(range(total_frames or int(1e9)), desc="MTMC"):
        all_done = True

        for cam in cameras:
            cap   = caps[cam]
            frame = None

            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    continue
                all_done = False
            else:
                if f_id in detections[cam]:
                    all_done = False

            frame_dets = detections[cam].get(f_id, [])
            results    = trackers[cam].track(frame_dets, f_id, frame=frame)

            # Buffer results (1-indexed frame to match MOT format)
            for tid, bbox in results:
                x, y, w, h = bbox
                all_results[cam].append((f_id + 1, tid, x, y, w, h))
                id_cameras[tid].add(cam)

        if all_done:
            break

    # ------------------------------------------------------------------
    # 5.  Release video captures
    # ------------------------------------------------------------------
    for cam in cameras:
        if caps[cam] is not None:
            caps[cam].release()

    # ------------------------------------------------------------------
    # 6.  Prune single-camera tracks
    # ------------------------------------------------------------------
    valid_ids = {gid for gid, cams in id_cameras.items() if len(cams) >= 2}
    n_pruned  = len(id_cameras) - len(valid_ids)
    print(f"\n  Global IDs total   : {global_id_mgr.current}")
    print(f"  Single-camera (pruned): {n_pruned}")
    print(f"  Multi-camera (kept)   : {len(valid_ids)}")

    # ------------------------------------------------------------------
    # 7.  Write filtered .txt results
    # ------------------------------------------------------------------
    for cam in cameras:
        cam_out  = os.path.join(out_dir, cam)
        txt_path = os.path.join(cam_out, "mtmc_sort_reid.txt")
        with open(txt_path, "w") as f:
            for (frame_id, tid, x, y, w, h) in all_results[cam]:
                if tid in valid_ids:
                    f.write(f"{frame_id},{tid},{x},{y},{w},{h},1,-1,-1,-1\n")

    print(f"\n  Results written to {out_dir}/")

    # ------------------------------------------------------------------
    # 8.  Per-camera evaluation against GT (optional)
    # ------------------------------------------------------------------
    if args.eval:
        if load_gt_tracks is None or evaluate_tracking is None:
            print("\n  [eval] Skipped – evaluation module not importable.")
        else:
            print(f"\n{'='*55}")
            print(f"  MTMC Evaluation  (global IDs, multi-camera pruned)")
            print(f"{'='*55}")

            import csv as _csv
            cam_metrics = {}

            for cam in cameras:
                gt_path = os.path.join(args.scenario_dir, cam, "gt", "gt.txt")
                if not os.path.exists(gt_path):
                    print(f"  [{cam}] No GT file at {gt_path} – skipped")
                    continue

                # Build pred_tracks: {frame_1idx: [(tid, (x,y,w,h)), ...]}
                pred_tracks = defaultdict(list)
                for (frame_1idx, tid, x, y, w, h) in all_results[cam]:
                    if tid in valid_ids:
                        pred_tracks[frame_1idx].append((tid, (x, y, w, h)))

                gt_tracks = load_gt_tracks(gt_path, train_frames=0)

                if not gt_tracks:
                    print(f"  [{cam}] GT is empty – skipped")
                    continue

                try:
                    metrics = evaluate_tracking(dict(pred_tracks), gt_tracks)
                    cam_metrics[cam] = metrics
                    print(f"  [{cam}]  HOTA={metrics['HOTA']:.4f}  IDF1={metrics['IDF1']:.4f}")
                except Exception as exc:
                    print(f"  [{cam}] Evaluation error: {exc}")

            if cam_metrics:
                mean_hota = float(np.mean([m["HOTA"] for m in cam_metrics.values()]))
                mean_idf1 = float(np.mean([m["IDF1"] for m in cam_metrics.values()]))
                print(f"  {'─'*45}")
                print(f"  Mean           HOTA={mean_hota:.4f}  IDF1={mean_idf1:.4f}")
                print(f"{'='*55}\n")

                if args.metrics_csv:
                    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
                    with open(args.metrics_csv, "w", newline="") as f:
                        writer = _csv.DictWriter(f, fieldnames=["camera", "HOTA", "IDF1"])
                        writer.writeheader()
                        for cam, m in cam_metrics.items():
                            writer.writerow({"camera": cam, "HOTA": m["HOTA"], "IDF1": m["IDF1"]})
                        writer.writerow({"camera": "mean", "HOTA": mean_hota, "IDF1": mean_idf1})
                    print(f"  Metrics saved → {args.metrics_csv}")

                return {"HOTA": mean_hota, "IDF1": mean_idf1}

    # ------------------------------------------------------------------
    # 10.  Optional video rendering (per-camera + montage)
    # ------------------------------------------------------------------
    if args.write_video or args.montage:
        render_outputs(args, cameras, all_results, valid_ids,
                       timestamps, cam_fps, out_dir)

    return None


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(args):
    """
    Sweep over all combinations of --gs_* parameter lists, run run_mtmc for
    each, and report the best configuration by mean HOTA.

    Parameters not supplied via --gs_* keep their fixed values from the
    corresponding base flags (e.g. --match_thr).

    Output
    ------
    A CSV file (--gs_csv or <out_dir>/gs_results.csv) with one row per
    combination, plus a summary of the best config printed to stdout.
    """
    import copy
    import itertools
    import contextlib
    import csv as _csv

    scenario_name = os.path.basename(args.scenario_dir.rstrip("/"))
    base_out = args.out_dir or os.path.join(SCRIPT_DIR, "results", scenario_name)
    gs_csv_path = args.gs_csv or os.path.join(base_out, "gs_results.csv")
    os.makedirs(base_out, exist_ok=True)

    # Parameter grid: name → list of values to try
    # If --gs_<param> was not given, use the single fixed value from base args.
    PARAM_FIELDS = [
        ("match_thr",       "gs_match_thr",       float),
        ("lookback",        "gs_lookback",         float),
        ("geo_weight",      "gs_geo_weight",       float),
        ("max_speed",       "gs_max_speed",        float),
        ("geo_sigma",       "gs_geo_sigma",        float),
        ("velocity_window", "gs_velocity_window",  int),
        ("iou_thr",         "gs_iou_thr",          float),
        ("max_age",         "gs_max_age",           int),
        ("min_hits",        "gs_min_hits",          int),
    ]

    grid = {}
    for param, gs_attr, _ in PARAM_FIELDS:
        values = getattr(args, gs_attr, None)
        grid[param] = values if values else [getattr(args, param)]

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n      = len(combos)

    print(f"\n{'='*60}")
    print(f"  Grid search: {n} combinations over {scenario_name}")
    for k in keys:
        if len(grid[k]) > 1:
            print(f"    {k}: {grid[k]}")
    print(f"  Results → {gs_csv_path}")
    print(f"{'='*60}\n")

    csv_fields = keys + ["HOTA", "IDF1"]
    rows = []
    best_row = None

    with open(gs_csv_path, "w", newline="") as fcsv:
        writer = _csv.DictWriter(fcsv, fieldnames=csv_fields)
        writer.writeheader()

        for i, combo in enumerate(combos):
            # Build a patched args copy for this combination
            run_args = copy.deepcopy(args)
            desc_parts = []
            for param, value in zip(keys, combo):
                setattr(run_args, param, value)
                if len(grid[param]) > 1:
                    desc_parts.append(f"{param}={value}")

            # Write to an isolated subdirectory so runs don't clobber each other
            combo_tag = "_".join(desc_parts) or "default"
            run_args.out_dir    = os.path.join(base_out, "gs", combo_tag)
            run_args.eval       = True
            run_args.write_video = False
            run_args.montage    = False
            run_args.metrics_csv = None   # handled by grid_search itself
            os.makedirs(run_args.out_dir, exist_ok=True)

            print(f"  [{i+1}/{n}] {combo_tag or 'default'}", end=" ... ", flush=True)

            # Suppress per-run chatter
            with open(os.devnull, "w") as devnull, \
                 contextlib.redirect_stdout(devnull):
                result = run_mtmc(run_args)

            if result is None:
                print("no GT – skipped")
                continue

            hota, idf1 = result["HOTA"], result["IDF1"]
            print(f"HOTA={hota:.4f}  IDF1={idf1:.4f}")

            row = dict(zip(keys, combo))
            row["HOTA"] = hota
            row["IDF1"]  = idf1
            rows.append(row)
            writer.writerow(row)
            fcsv.flush()

            if best_row is None or hota > best_row["HOTA"]:
                best_row = row

    print(f"\n{'='*60}")
    print(f"  Grid search complete ({len(rows)}/{n} runs evaluated)")
    if best_row:
        print(f"  Best config (HOTA={best_row['HOTA']:.4f}  IDF1={best_row['IDF1']:.4f}):")
        for k in keys:
            if len(grid[k]) > 1:
                print(f"    --{k} {best_row[k]}")
    print(f"  Full results → {gs_csv_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.grid_search:
        grid_search(args)
    else:
        run_mtmc(args)