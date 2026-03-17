"""
sanity_check_homography.py
--------------------------
Projects the ROI mask of one or more cameras into world coordinates using
the calibration homography, then plots the results as a bird's-eye-view map.

Usage
-----
    python sanity_check_homography.py \
        --scenario_dir /path/to/AI_CITY_CHALLENGE_2022_TRAIN/train/S01 \
        --out sanity_homography.png

How it works
------------
For each camera:
  1. Load the ROI mask (roi.jpg  – white = valid region).
  2. Extract the boundary of the white region.
  3. Apply the calibration homography H to each boundary pixel:
         [wx, wy, w] = H @ [u, v, 1]^T
         world_x, world_y = wx/w, wy/w
  4. Plot the resulting polygon in world-coordinate space.

A well-calibrated homography should produce non-overlapping, spatially
consistent footprints that reflect the physical camera layout.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – saves to file instead of showing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo_utils import load_homography, pixel_to_geo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_cameras(scenario_dir):
    return sorted([
        d for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith("c")
    ])


def roi_boundary_pixels(roi_path, step=10):
    """
    Return a list of (u, v) pixel coordinates that lie on the *boundary*
    of the white region in the ROI mask.

    `step` controls the sampling density (every `step`-th contour point).
    """
    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    pts = contour[::step, 0, :]        # shape (N, 2) → (u, v)
    return pts.tolist()


def homography_denominator(H, u, v):
    """Return the perspective denominator w = H[2] · [u, v, 1]."""
    return H[2, 0] * u + H[2, 1] * v + H[2, 2]


def project_pixels(H, pixel_pts, min_denom=0.05):
    """
    Apply homography H to a list of (u, v) pixels; return world (x, y) list.

    Points where |w| < min_denom are near the homography's vanishing line
    (horizon) and project to near-infinity – they are silently skipped.
    """
    world_pts = []
    for u, v in pixel_pts:
        w = homography_denominator(H, float(u), float(v))
        if abs(w) < min_denom:
            continue                # near-singular, skip
        geo = pixel_to_geo(H, float(u), float(v))
        if geo is not None:
            world_pts.append(geo)
    return world_pts


def remove_outliers(pts, n_sigma=3.0):
    """Remove points more than n_sigma standard deviations from the median."""
    if len(pts) < 5:
        return pts
    arr = np.array(pts)
    med = np.median(arr, axis=0)
    std = np.std(arr, axis=0) + 1e-9
    mask = np.all(np.abs(arr - med) < n_sigma * std, axis=1)
    return arr[mask].tolist()


def project_grid(H, mask_path, grid_step=30, min_denom=0.05):
    """
    Project a regular grid of *white* pixels from the ROI mask.
    Useful for visualising the density and distortion of the projection.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape
    pts = []
    for v in range(0, h, grid_step):
        for u in range(0, w, grid_step):
            if binary[v, u] > 0:
                denom = homography_denominator(H, float(u), float(v))
                if abs(denom) < min_denom:
                    continue
                geo = pixel_to_geo(H, float(u), float(v))
                if geo is not None:
                    pts.append(geo)
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def parse_args():
    p = argparse.ArgumentParser(description="Homography sanity-check: project ROI to world coords")
    p.add_argument("--scenario_dir",
                   default="../AI_CITY_CHALLENGE_2022_TRAIN/train/S01",
                   help="Path to scenario directory containing camera sub-folders")
    p.add_argument("--out", default="sanity_homography.png",
                   help="Output image path")
    p.add_argument("--step", default=5, type=int,
                   help="Boundary contour sampling step (lower = more points)")
    p.add_argument("--grid_step", default=25, type=int,
                   help="Grid sampling step for interior points (0 = skip)")
    return p.parse_args()


def main():
    args = parse_args()

    cameras = discover_cameras(args.scenario_dir)
    if not cameras:
        print(f"No camera directories found in {args.scenario_dir}")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_world  = axes[0]
    ax_pixels = axes[1]

    legend_patches = []

    # Collect all valid projections first so we can build a shared metre origin
    all_gps_pts = []    # (lat, lon) from every camera
    cam_data = {}       # cam -> {"boundary_px", "boundary_gps", "grid_gps"}

    # ── Pass 1: collect GPS data for every camera ────────────────────────
    for i, cam in enumerate(cameras):
        cal_path = os.path.join(args.scenario_dir, cam, "calibration.txt")
        roi_path = os.path.join(args.scenario_dir, cam, "roi.jpg")

        H = load_homography(cal_path)
        if H is None:
            print(f"  [{cam}] SKIP – calibration not found at {cal_path}")
            continue
        if not os.path.exists(roi_path):
            print(f"  [{cam}] SKIP – roi.jpg not found at {roi_path}")
            continue

        boundary_px = roi_boundary_pixels(roi_path, step=args.step)
        if not boundary_px:
            print(f"  [{cam}] SKIP – could not extract ROI boundary")
            continue

        boundary_gps = project_pixels(H, boundary_px)
        boundary_gps = remove_outliers(boundary_gps)
        if not boundary_gps:
            print(f"  [{cam}] SKIP – homography produced no valid projections")
            continue

        grid_gps = []
        if args.grid_step > 0:
            grid_gps = project_grid(H, roi_path, grid_step=args.grid_step)
            grid_gps = remove_outliers(grid_gps)

        cam_data[cam] = {
            "color":       COLORS[i % len(COLORS)],
            "boundary_px": boundary_px,
            "boundary_gps": boundary_gps,
            "grid_gps":    grid_gps,
        }
        all_gps_pts.extend(boundary_gps)
        all_gps_pts.extend(grid_gps)

    if not all_gps_pts:
        print("No valid projections found.")
        return

    # ── Build a local metric origin (centroid in GPS, project to East-North metres)
    all_arr = np.array(all_gps_pts)
    lat0 = np.mean(all_arr[:, 0])
    lon0 = np.mean(all_arr[:, 1])
    M_lat = 111_000.0                             # metres per degree latitude
    M_lon = 111_000.0 * np.cos(np.radians(lat0)) # metres per degree longitude

    def gps_to_metres(lat, lon):
        return (lon - lon0) * M_lon, (lat - lat0) * M_lat   # (East, North)

    # ── Pass 2: plot ─────────────────────────────────────────────────────
    for cam, data in cam_data.items():
        color       = data["color"]
        boundary_px = data["boundary_px"]
        boundary_gps= data["boundary_gps"]
        grid_gps    = data["grid_gps"]

        # Convert GPS → local metres
        bx, by = zip(*[gps_to_metres(lat, lon) for lat, lon in boundary_gps])

        ax_world.scatter(bx, by, s=6, color=color, alpha=0.7)

        if grid_gps:
            gx, gy = zip(*[gps_to_metres(lat, lon) for lat, lon in grid_gps])
            ax_world.scatter(gx, gy, s=3, color=color, alpha=0.35)

        # Pixel-space boundary
        pu = [p[0] for p in boundary_px] + [boundary_px[0][0]]
        pv = [p[1] for p in boundary_px] + [boundary_px[0][1]]
        ax_pixels.plot(pu, pv, "-", color=color, linewidth=1.5, alpha=0.85)
        ax_pixels.fill(pu, pv, color=color, alpha=0.15)

        legend_patches.append(mpatches.Patch(color=color, label=cam))
        lat_arr = [p[0] for p in boundary_gps]
        lon_arr = [p[1] for p in boundary_gps]
        print(f"  [{cam}] OK – {len(boundary_gps)} pts  "
              f"GPS lat=[{min(lat_arr):.6f}, {max(lat_arr):.6f}]  "
              f"lon=[{min(lon_arr):.6f}, {max(lon_arr):.6f}]")

    # ── Formatting ────────────────────────────────────────────────────────
    ax_world.set_title(f"Camera ROI footprints – Local metric coords\n"
                       f"(GPS origin: {lat0:.5f}°N, {lon0:.5f}°E)")
    ax_world.set_xlabel("East (metres)")
    ax_world.set_ylabel("North (metres)")
    ax_world.set_aspect("equal")
    ax_world.grid(True, linestyle="--", alpha=0.4)
    ax_world.legend(handles=legend_patches, loc="best")

    ax_pixels.set_title("Camera ROI boundaries – Pixel space\n(before projection, for reference)")
    ax_pixels.set_xlabel("Pixel u")
    ax_pixels.set_ylabel("Pixel v")
    ax_pixels.invert_yaxis()
    ax_pixels.set_aspect("equal")
    ax_pixels.grid(True, linestyle="--", alpha=0.4)
    ax_pixels.legend(handles=legend_patches, loc="best")

    scenario = os.path.basename(args.scenario_dir.rstrip("/"))
    fig.suptitle(f"Homography sanity check – {scenario}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
