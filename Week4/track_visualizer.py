"""
track_visualizer.py
-------------------
Given a global car ID from offline MTMC output, generate a video showing:
  - Left:  map with GPS dot + fading trail (color-coded per camera)
  - Right: adaptive camera grid (only cameras currently seeing the car)

Usage:
    python track_visualizer.py \
        --out_dir  results/offline/S03 \
        --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
        --car_id 5 --output car_5.mp4

    # List available IDs first:
    python track_visualizer.py \
        --out_dir results/offline/S03 \
        --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
        --list_ids
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.geo_utils import load_homography, load_timestamps, pixel_to_geo, bbox_foot_point

try:
    import contextily as ctx
    from pyproj import Transformer
    _HAVE_MAP = True
except ImportError:
    _HAVE_MAP = False
    print("[warn] contextily/pyproj not installed – map will be blank")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── layout ───────────────────────────────────────────────────────────────────
MAP_W, MAP_H   = 640, 640   # map panel (square)
CAM_W, CAM_H   = 480, 270   # each camera panel
GRID_COLS      = 2          # max columns in camera grid
TRAIL_LEN      = 80         # GPS history points to show
GPS_TOLERANCE  = 0.7        # seconds: max gap to consider a camera "active"
OUTPUT_FPS     = 10.0       # output video framerate
MAP_PAD_DEG    = 0.0006     # padding around GPS bounding box

# ── colour palette (BGR) for cameras ─────────────────────────────────────────
CAM_PALETTE_BGR = [
    (75,  25, 230),   # blue-ish red
    (75, 180,  60),   # green
    (216, 99,  67),   # orange-blue
    (49, 130, 245),   # orange
    (180, 30, 145),   # purple
    (244, 212,  66),  # cyan-yellow
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _bgr(palette_idx: int) -> Tuple[int, int, int]:
    return CAM_PALETTE_BGR[palette_idx % len(CAM_PALETTE_BGR)]


def _lerp_color(c0, c1, alpha):
    return tuple(int(c0[i] + alpha * (c1[i] - c0[i])) for i in range(3))


# ── data loading ─────────────────────────────────────────────────────────────

def discover_cameras(scenario_dir: str) -> List[str]:
    return sorted(
        d for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith("c")
    )


def video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 10.0


def load_scenario(out_dir: str, scenario_dir: str) -> Dict:
    """
    Load per-camera metadata and detection records.

    Returns
    -------
    cam_info : {
        cam: {
            fps, offset, video_path, H,
            detections: {gid: [(frame_0idx, bbox_xywh)]}
        }
    }
    """
    cameras = discover_cameras(scenario_dir)

    scenario_name = os.path.basename(scenario_dir.rstrip(os.sep))
    ts_path = os.path.normpath(
        os.path.join(scenario_dir, "..", "..", "cam_timestamp",
                     f"{scenario_name}.txt")
    )
    timestamps = load_timestamps(ts_path) if os.path.exists(ts_path) else {}

    cam_info = {}
    for cam in cameras:
        cam_dir   = os.path.join(scenario_dir, cam)
        txt_path  = os.path.join(out_dir, cam, "offline_mtmc.txt")
        video_path = os.path.join(cam_dir, "vdo.avi")

        if not os.path.exists(txt_path) or not os.path.exists(video_path):
            continue

        fps    = video_fps(video_path)
        offset = timestamps.get(cam, 0.0)
        H      = load_homography(os.path.join(cam_dir, "calibration.txt"))

        detections: Dict[int, List] = defaultdict(list)
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame_1idx = int(parts[0])
                gid        = int(parts[1])
                bbox       = np.array([float(parts[2]), float(parts[3]),
                                       float(parts[4]), float(parts[5])])
                detections[gid].append((frame_1idx - 1, bbox))

        cam_info[cam] = {
            "fps": fps, "offset": offset,
            "video_path": video_path,
            "H": H,
            "detections": dict(detections),
        }

    return cam_info


def build_timeline(cam_info: Dict, car_id: int) -> Dict[str, List]:
    """
    For a given car_id, return per-camera timeline sorted by absolute time:
        {cam: [(abs_time, frame_0idx, bbox, gps_or_None)]}
    """
    timeline = {}
    for cam, info in cam_info.items():
        if car_id not in info["detections"]:
            continue
        fps, offset, H = info["fps"], info["offset"], info["H"]
        records = []
        for frame_0idx, bbox in sorted(info["detections"][car_id]):
            t   = offset + frame_0idx / fps
            gps = None
            if H is not None:
                u, v = bbox_foot_point(bbox)
                gps  = pixel_to_geo(H, u, v)
            records.append((t, frame_0idx, bbox, gps))
        if records:
            timeline[cam] = records
    return timeline


# ── map rendering ─────────────────────────────────────────────────────────────

def prerender_basemap(timeline: Dict) -> Tuple:
    """
    Download and rasterise an OSM tile covering all GPS positions.

    Returns (base_bgr, extent_merc, transformer) or (None, None, None).
    extent_merc = (x_min, x_max, y_min, y_max) in Web Mercator.
    """
    if not _HAVE_MAP:
        return None, None, None

    lats, lons = [], []
    for recs in timeline.values():
        for _, _, _, gps in recs:
            if gps is not None:
                lats.append(gps[0]); lons.append(gps[1])

    if not lats:
        return None, None, None

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lat_min, lat_max = min(lats) - MAP_PAD_DEG, max(lats) + MAP_PAD_DEG
    lon_min, lon_max = min(lons) - MAP_PAD_DEG, max(lons) + MAP_PAD_DEG
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    print("  Downloading basemap tile …", end=" ", flush=True)
    fig, ax = plt.subplots(figsize=(MAP_W / 100, MAP_H / 100), dpi=100)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom="auto", attribution_size=5)
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rgb = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return base_bgr, (x_min, x_max, y_min, y_max), transformer


def gps_to_px(lat: float, lon: float, extent, transformer) -> Tuple[int, int]:
    x_min, x_max, y_min, y_max = extent
    mx, my = transformer.transform(lon, lat)
    col = int((mx - x_min) / max(x_max - x_min, 1e-9) * MAP_W)
    row = int((1.0 - (my - y_min) / max(y_max - y_min, 1e-9)) * MAP_H)
    return col, row


def render_map_frame(base_bgr, extent, transformer,
                     trail, current_entry, cam_order):
    """
    Parameters
    ----------
    trail        : list of (lat, lon, cam_name) – older → newer
    current_entry: (lat, lon, cam_name) or None
    cam_order    : {cam: palette_index}
    """
    img = base_bgr.copy()

    # ── fading trail ─────────────────────────────────────────────────────────
    for i, (lat, lon, cam) in enumerate(trail):
        alpha  = (i + 1) / max(len(trail), 1)
        pidx   = cam_order.get(cam, 0)
        color  = _bgr(pidx)
        faded  = _lerp_color((30, 30, 30), color, alpha)
        col, row = gps_to_px(lat, lon, extent, transformer)
        if 0 <= col < MAP_W and 0 <= row < MAP_H:
            r = max(2, int(5 * alpha))
            cv2.circle(img, (col, row), r, faded, -1)

    # ── current dot ──────────────────────────────────────────────────────────
    if current_entry is not None:
        lat, lon, cam = current_entry
        pidx  = cam_order.get(cam, 0)
        color = _bgr(pidx)
        col, row = gps_to_px(lat, lon, extent, transformer)
        if 0 <= col < MAP_W and 0 <= row < MAP_H:
            cv2.circle(img, (col, row), 13, (255, 255, 255), -1)
            cv2.circle(img, (col, row), 11, color, -1)
            cv2.putText(img, cam, (col + 15, row + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, cam, (col + 15, row + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    return img


# ── camera panel rendering ────────────────────────────────────────────────────

class VideoReader:
    """Lazy video reader with sequential-read optimisation."""

    def __init__(self, path: str):
        self._path  = path
        self._cap   = None
        self._prev  = -2   # last frame index actually read

    def _open(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._path)

    def read(self, frame_0idx: int) -> Optional[np.ndarray]:
        self._open()
        if frame_0idx != self._prev + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_0idx)
        ok, frame = self._cap.read()
        self._prev = frame_0idx if ok else -2
        return frame if ok else None

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None


def cam_panel(frame, bbox, cam: str, palette_idx: int, car_id: int) -> np.ndarray:
    """Draw bbox on frame and resize to CAM_W × CAM_H."""
    color = _bgr(palette_idx)

    if frame is not None:
        out = frame.copy()
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        cv2.putText(out, f"ID {car_id}", (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(out, f"ID {car_id}", (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        panel = cv2.resize(out, (CAM_W, CAM_H))
    else:
        panel = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    # Label bar at top
    cv2.rectangle(panel, (0, 0), (CAM_W, 30), tuple(c // 3 for c in color), -1)
    cv2.putText(panel, cam, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return panel


def build_grid(panels: List[np.ndarray], target_h: int) -> np.ndarray:
    """Arrange panels in ≤GRID_COLS columns, padded to target_h."""
    if not panels:
        return np.zeros((target_h, CAM_W, 3), dtype=np.uint8)

    n_cols = min(GRID_COLS, len(panels))
    n_rows = (len(panels) + n_cols - 1) // n_cols
    grid_h = n_rows * CAM_H
    grid_w = n_cols * CAM_W

    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, n_cols)
        grid[r * CAM_H:(r + 1) * CAM_H, c * CAM_W:(c + 1) * CAM_W] = p

    if grid_h < target_h:
        grid = np.vstack([grid,
                          np.zeros((target_h - grid_h, grid_w, 3), np.uint8)])
    return grid[:target_h]


# ── main render loop ──────────────────────────────────────────────────────────

def render_car_video(cam_info: Dict, car_id: int, output_path: str,
                     out_fps: float = OUTPUT_FPS):

    print(f"\n  Building timeline for car ID {car_id} …")
    timeline = build_timeline(cam_info, car_id)

    if not timeline:
        print(f"  Car ID {car_id} not found in any camera output")
        return

    cameras   = sorted(timeline.keys())
    cam_order = {cam: i for i, cam in enumerate(cameras)}
    print(f"  Cameras: {cameras}")

    base_bgr, extent, merc_tf = prerender_basemap(timeline)

    # ── time range ────────────────────────────────────────────────────────────
    all_times = [t for recs in timeline.values() for t, *_ in recs]
    t_start   = min(all_times) - 0.3
    t_end     = max(all_times) + 0.3
    n_frames  = int((t_end - t_start) * out_fps) + 1
    print(f"  [{t_start:.2f} s → {t_end:.2f} s]  {n_frames} output frames")

    # ── peak simultaneous cameras (to fix grid size) ──────────────────────────
    peak = 1
    for fi in range(n_frames):
        t = t_start + fi / out_fps
        n_active = sum(
            1 for recs in timeline.values()
            if any(abs(rec[0] - t) <= GPS_TOLERANCE for rec in recs)
        )
        peak = max(peak, n_active)

    n_grid_cols = min(GRID_COLS, peak)
    n_grid_rows = (peak + n_grid_cols - 1) // n_grid_cols
    GRID_W = n_grid_cols * CAM_W
    OUT_H  = max(MAP_H, n_grid_rows * CAM_H)
    OUT_W  = MAP_W + GRID_W
    print(f"  Output: {OUT_W}×{OUT_H}  (peak {peak} cameras simultaneously)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (OUT_W, OUT_H))

    readers = {cam: VideoReader(cam_info[cam]["video_path"]) for cam in cameras}
    trail: List[Tuple] = []

    for fi in range(n_frames):
        t = t_start + fi / out_fps

        # ── find active cameras ───────────────────────────────────────────────
        active = {}   # cam → (abs_time, frame_0idx, bbox, gps)
        for cam, recs in timeline.items():
            best, best_dt = None, float("inf")
            for rec in recs:
                dt = abs(rec[0] - t)
                if dt < best_dt and dt <= GPS_TOLERANCE:
                    best, best_dt = rec, dt
            if best is not None:
                active[cam] = best

        # ── GPS + trail ───────────────────────────────────────────────────────
        current_entry = None
        if active:
            src_cam = min(active, key=lambda c: abs(active[c][0] - t))
            gps = active[src_cam][3]
            if gps is not None:
                current_entry = (gps[0], gps[1], src_cam)
                trail.append(current_entry)
                if len(trail) > TRAIL_LEN:
                    trail.pop(0)

        # ── map panel ─────────────────────────────────────────────────────────
        if base_bgr is not None and extent is not None:
            map_frame = render_map_frame(
                base_bgr, extent, merc_tf, trail, current_entry, cam_order)
        else:
            map_frame = np.full((MAP_H, MAP_W, 3), 20, dtype=np.uint8)
            cv2.putText(map_frame, "No map", (MAP_W // 2 - 50, MAP_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

        # Pad map to OUT_H
        if map_frame.shape[0] < OUT_H:
            pad = np.zeros((OUT_H - map_frame.shape[0], MAP_W, 3), np.uint8)
            map_frame = np.vstack([map_frame, pad])
        map_frame = map_frame[:OUT_H]

        # ── camera panels ─────────────────────────────────────────────────────
        panels = []
        for cam in cameras:
            if cam not in active:
                continue
            _, frame_0idx, bbox, _ = active[cam]
            frame   = readers[cam].read(frame_0idx)
            panel   = cam_panel(frame, bbox, cam, cam_order[cam], car_id)
            panels.append(panel)

        grid = build_grid(panels, OUT_H)

        # Pad/trim grid width
        if grid.shape[1] < GRID_W:
            pad = np.zeros((OUT_H, GRID_W - grid.shape[1], 3), np.uint8)
            grid = np.hstack([grid, pad])
        grid = grid[:, :GRID_W]

        # ── compose ───────────────────────────────────────────────────────────
        out_frame = np.hstack([map_frame, grid])

        # Info overlays
        cv2.putText(out_frame, f"Car ID: {car_id}",
                    (8, OUT_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(out_frame, f"t = {t:.2f} s",
                    (MAP_W + 8, OUT_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # Camera legend on map
        for i, cam in enumerate(cameras):
            cy = 28 + i * 22
            color = _bgr(cam_order[cam])
            cv2.rectangle(out_frame, (8, cy - 14), (20, cy), color, -1)
            is_active = cam in active
            label = cam if is_active else f"{cam} (inactive)"
            text_color = (255, 255, 255) if is_active else (100, 100, 100)
            cv2.putText(out_frame, label, (26, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        writer.write(out_frame)

        if fi % 50 == 0:
            print(f"    frame {fi}/{n_frames}  t={t:.1f}s  "
                  f"active_cams={list(active.keys())}")

    writer.release()
    for r in readers.values():
        r.release()

    print(f"\n  Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\nLoading scenario data …")
    cam_info = load_scenario(args.out_dir, args.scenario_dir)

    if not cam_info:
        print("No data found (check --out_dir and --scenario_dir)")
        return

    if args.list_ids:
        all_ids: set = set()
        for info in cam_info.values():
            all_ids.update(info["detections"].keys())
        id_cam_count = defaultdict(int)
        for info in cam_info.values():
            for gid in info["detections"]:
                id_cam_count[gid] += 1
        print(f"\n  {'ID':>6}  {'cameras':>8}")
        print(f"  {'─'*6}  {'─'*8}")
        for gid in sorted(all_ids):
            print(f"  {gid:>6}  {id_cam_count[gid]:>8}")
        return

    if args.car_id is None:
        print("Provide --car_id N, or use --list_ids to see available IDs")
        return

    render_car_video(cam_info, args.car_id, args.output, out_fps=args.fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",      required=True,
                   help="Directory with per-camera offline_mtmc.txt files "
                        "(e.g. results/offline/S03)")
    p.add_argument("--scenario_dir", required=True,
                   help="Dataset scenario dir (e.g. .../train/S03)")
    p.add_argument("--car_id",       type=int, default=None,
                   help="Global car ID to visualize")
    p.add_argument("--output",       default="car_track.mp4",
                   help="Output video path")
    p.add_argument("--list_ids",     action="store_true",
                   help="List all available global IDs and exit")
    p.add_argument("--fps",          type=float, default=OUTPUT_FPS,
                   help="Output video FPS")
    main(p.parse_args())
