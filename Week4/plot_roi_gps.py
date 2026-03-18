"""
plot_roi_gps.py
---------------
Maps the ROI of each S03 camera from pixel space to GPS coordinates via the
inverse homography (pixel→GPS), then overlays all camera footprints on a
shared map.

A background tile is downloaded with contextily (OpenStreetMap) and
reprojected to WGS84 for display.

Usage:
    conda run -n flowformerpp python plot_roi_gps.py \
        [--scenario_dir PATH]  [--out map_roi.png]

Requirements:  numpy, opencv-python, matplotlib, contextily, pyproj
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── optional map background ─────────────────────────────────────────────────
try:
    import contextily as ctx
    from pyproj import Transformer
    _HAVE_CTX = True
except ImportError:
    _HAVE_CTX = False
    print("[warn] contextily / pyproj not available – plotting axes only")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette for cameras ──────────────────────────────────────────────
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4",
]


# ── homography helpers ───────────────────────────────────────────────────────

def load_homography(calib_path: str):
    """Return pixel→GPS homography (inverts the GPS→pixel matrix in file)."""
    if not os.path.exists(calib_path):
        return None
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Homography matrix:"):
                vals = line[len("Homography matrix:"):].strip()
                rows = vals.split(";")
                H = np.array([[float(v) for v in r.split()] for r in rows],
                              dtype=np.float64)
                if H.shape == (3, 3):
                    return np.linalg.inv(H)
    return None


def pixels_to_gps(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Project pixel coordinates (N×2, col=x, row=y) to GPS via homography.
    Returns (N×2) array with columns [lat, lon].
    """
    ones  = np.ones((len(pts), 1))
    hom   = (H @ np.hstack([pts, ones]).T).T   # N×3
    gps   = hom[:, :2] / hom[:, 2:3]          # divide by w
    return gps  # [lat, lon]


# ── ROI contour extraction ───────────────────────────────────────────────────

def roi_contour_pixels(roi_path: str, n_pts: int = 200,
                       erode_px: int = 40) -> np.ndarray:
    """
    Load roi.jpg, erode the mask by erode_px pixels to pull contour away from
    frame edges (where the homography extrapolates wildly), extract the outer
    contour, and return up to n_pts evenly-spaced contour points as (N×2)
    [x, y] pixel arrays.
    """
    mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(roi_path)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        binary = cv2.erode(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError(f"No contours found in {roi_path}")

    # largest contour
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.reshape(-1, 2).astype(float)   # [x, y]

    # subsample to n_pts evenly spaced points
    idx = np.linspace(0, len(cnt) - 1, n_pts, dtype=int)
    return cnt[idx]


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    scenario_dir = args.scenario_dir
    cameras      = sorted(
        d for d in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith("c")
    )

    cam_polygons = {}   # cam → (N×2) [lat, lon]

    for cam in cameras:
        cam_dir  = os.path.join(scenario_dir, cam)
        calib    = os.path.join(cam_dir, "calibration.txt")
        roi_path = os.path.join(cam_dir, "roi.jpg")

        H = load_homography(calib)
        if H is None:
            print(f"  [{cam}] no homography – skipped")
            continue
        if not os.path.exists(roi_path):
            print(f"  [{cam}] no roi.jpg – skipped")
            continue

        try:
            pts_px  = roi_contour_pixels(roi_path, n_pts=300)
            pts_gps = pixels_to_gps(H, pts_px)          # [lat, lon]

            # Filter outliers: keep points within 0.005° of per-camera median
            # (homography can extrapolate wildly for pixels near frame edges)
            med_lat = np.median(pts_gps[:, 0])
            med_lon = np.median(pts_gps[:, 1])
            dist = np.sqrt((pts_gps[:, 0] - med_lat)**2 +
                           (pts_gps[:, 1] - med_lon)**2)
            pts_gps = pts_gps[dist < 0.002]

            if len(pts_gps) < 3:
                print(f"  [{cam}] too few inliers after outlier filter – skipped")
                continue

            cam_polygons[cam] = pts_gps
            lat_c = pts_gps[:, 0].mean()
            lon_c = pts_gps[:, 1].mean()
            print(f"  [{cam}] {len(pts_gps)} inlier pts, "
                  f"centroid ≈ ({lat_c:.5f}, {lon_c:.5f})")
        except Exception as e:
            print(f"  [{cam}] error: {e}")
            continue

    if not cam_polygons:
        print("No cameras with valid data – abort")
        return

    # ── bounding box of all GPS points ──────────────────────────────────────
    all_lats = np.concatenate([v[:, 0] for v in cam_polygons.values()])
    all_lons = np.concatenate([v[:, 1] for v in cam_polygons.values()])
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()
    pad = 0.0003
    lat_min -= pad; lat_max += pad
    lon_min -= pad; lon_max += pad

    print(f"\n  GPS bounding box: lat [{lat_min:.5f}, {lat_max:.5f}]  "
          f"lon [{lon_min:.5f}, {lon_max:.5f}]")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)

    # ── map background ───────────────────────────────────────────────────────
    if _HAVE_CTX and not args.no_map:
        # Convert WGS84 bbox to Web Mercator (EPSG:3857) for contextily
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        try:
            ctx.add_basemap(ax, crs="EPSG:3857",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom="auto", attribution_size=6)
            print("  Map tile downloaded (OpenStreetMap)")
        except Exception as e:
            print(f"  [warn] tile download failed: {e}")

        # project ROI polygons to Web Mercator for plotting
        def to_mercator(gps_pts):
            lons = gps_pts[:, 1]
            lats = gps_pts[:, 0]
            xs, ys = transformer.transform(lons, lats)
            return np.column_stack([xs, ys])

        for i, (cam, gps) in enumerate(cam_polygons.items()):
            color = PALETTE[i % len(PALETTE)]
            merc  = to_mercator(gps)
            poly  = plt.Polygon(merc, closed=True,
                                facecolor=color, alpha=0.35,
                                edgecolor=color, linewidth=2)
            ax.add_patch(poly)
            cx, cy = merc[:, 0].mean(), merc[:, 1].mean()
            ax.text(cx, cy, cam, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8))

        ax.set_xlabel("Easting (m)  –  Web Mercator")
        ax.set_ylabel("Northing (m)  –  Web Mercator")

    else:
        # plain lat/lon axes, no tile
        for i, (cam, gps) in enumerate(cam_polygons.items()):
            color = PALETTE[i % len(PALETTE)]
            lons  = gps[:, 1]
            lats  = gps[:, 0]
            ax.fill(lons, lats, color=color, alpha=0.35)
            ax.plot(lons, lats, color=color, linewidth=2)
            ax.text(lons.mean(), lats.mean(), cam,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8))

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.invert_xaxis()   # west on left

    # ── legend ───────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=cam)
               for i, cam in enumerate(cam_polygons)]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    sequence = os.path.basename(scenario_dir.rstrip("/"))
    ax.set_title(f"Camera ROIs in GPS space  –  {sequence}", fontsize=13)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"\n  Saved → {args.out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario_dir", default=os.path.normpath(
        os.path.join(SCRIPT_DIR, "..",
                     "AI_CITY_CHALLENGE_2022_TRAIN", "train", "S03")))
    p.add_argument("--out", default="map_roi_s03.png",
                   help="Output image path")
    p.add_argument("--no_map", action="store_true",
                   help="Skip tile download, plot plain lat/lon axes")
    main(p.parse_args())
