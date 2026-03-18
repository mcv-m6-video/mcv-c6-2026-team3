"""
aic_eval.py
-----------
AI City Challenge official evaluation logic, adapted to run without pytrec_eval
(which is incompatible with Python 3.8).

The evaluation mirrors the compare_dataframes_mtmc() function in the official
eval/eval.py but is callable directly from Python rather than via CLI.

Metric: IDF1 / IDP / IDR  (computed with motmetrics across all cameras jointly)

Input format (AIC single-file):
    CameraId  Id  FrameId  X  Y  Width  Height  Xworld  Yworld
    Space/tab/comma separated, no header.

Camera ID convention: numeric, e.g. c001 -> 1, c015 -> 15.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    import motmetrics as mm
    _HAVE_DEPS = True
except ImportError:
    _HAVE_DEPS = False


AIC_COLUMNS = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]

# S01 -> cameras 1-5, S03 -> 10-15, S04 -> 16-40
SEQUENCE_CAMERAS = {
    "S01": list(range(1, 6)),
    "S03": list(range(10, 16)),
    "S04": list(range(16, 41)),
}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_aic_gt(gt_path: str, cam_ids: Optional[List[int]] = None) -> "pd.DataFrame":
    """Load the AIC ground truth file and optionally filter to specific cameras."""
    if not _HAVE_DEPS:
        raise ImportError("pandas and motmetrics are required for AIC evaluation")
    df = pd.read_csv(gt_path, sep=r"\s+|,", header=None, names=AIC_COLUMNS, engine="python")
    if cam_ids:
        df = df[df["CameraId"].isin(cam_ids)].copy()
    return df


def load_aic_gt_for_sequence(gt_path: str, sequence: str) -> "pd.DataFrame":
    """Load GT filtered to a single sequence (e.g. 'S03')."""
    cam_ids = SEQUENCE_CAMERAS.get(sequence.upper())
    if cam_ids is None:
        raise ValueError(f"Unknown sequence '{sequence}'. Known: {list(SEQUENCE_CAMERAS)}")
    return load_aic_gt(gt_path, cam_ids=cam_ids)


# ---------------------------------------------------------------------------
# Prediction format conversion
# ---------------------------------------------------------------------------

def cam_name_to_id(cam: str) -> int:
    """Convert camera directory name to numeric ID: 'c010' -> 10."""
    return int(cam.lstrip("c").lstrip("0") or "0")


def mot_file_to_aic_df(txt_path: str, cam_id: int) -> "pd.DataFrame":
    """
    Read a per-camera MOTChallenge result file and return an AIC-format DataFrame.

    MOTChallenge format (comma separated):
        frame_id, track_id, x, y, w, h, conf, -1, -1, -1
    or:
        frame_id, track_id, x, y, w, h   (short form)
    """
    if not os.path.exists(txt_path):
        return pd.DataFrame(columns=AIC_COLUMNS)

    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = int(float(parts[2])), int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
            rows.append([cam_id, track_id, frame_id, x, y, w, h, -1, -1])

    return pd.DataFrame(rows, columns=AIC_COLUMNS)


def merge_cam_outputs(
    out_dir: str,
    cameras: List[str],
    filename: str = "mtmc_sort_reid.txt",
) -> "pd.DataFrame":
    """
    Merge per-camera MOTChallenge output files into a single AIC-format DataFrame.

    Parameters
    ----------
    out_dir   : root output directory containing <cam>/<filename> files
    cameras   : list of camera names, e.g. ['c010', 'c011', ...]
    filename  : name of the per-camera result file
    """
    dfs = []
    for cam in cameras:
        path = os.path.join(out_dir, cam, filename)
        cam_id = cam_name_to_id(cam)
        df = mot_file_to_aic_df(path, cam_id)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=AIC_COLUMNS)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Core evaluation (mirrors compare_dataframes_mtmc from eval.py)
# ---------------------------------------------------------------------------

def _remove_single_cam(pred: "pd.DataFrame") -> "pd.DataFrame":
    """Remove predicted IDs that appear in only one camera."""
    cnt = pred[["CameraId", "Id"]].drop_duplicates()[["Id"]].groupby("Id").size()
    keep = cnt[cnt > 1].index
    return pred[pred["Id"].isin(keep)]


def _remove_repetition(pred: "pd.DataFrame") -> "pd.DataFrame":
    """Remove duplicate (cam, id, frame) entries."""
    return pred.drop_duplicates(subset=["CameraId", "Id", "FrameId"], keep="first")


def run_aic_eval(
    pred: "pd.DataFrame",
    gt: "pd.DataFrame",
    filter_single_cam: bool = True,
) -> Dict:
    """
    Run the AIC multi-camera IDF1 evaluation.

    Parameters
    ----------
    pred              : AIC-format prediction DataFrame
    gt                : AIC-format ground-truth DataFrame (already filtered to desired cameras)
    filter_single_cam : apply removeOutliersSingleCam to predictions (default True)

    Returns
    -------
    dict with keys: idf1, idp, idr  (values in [0, 1])
    """
    if not _HAVE_DEPS:
        raise ImportError("pandas and motmetrics are required for AIC evaluation")

    # Filter predictions
    if filter_single_cam:
        pred = _remove_single_cam(pred)
    pred = _remove_repetition(pred)

    gtds, tsds = [], []
    gt_cams = sorted(gt["CameraId"].unique())
    ts_cams = set(pred["CameraId"].unique()) if not pred.empty else set()
    max_frame = 0

    for cid in gt_cams:
        gtd = gt[gt["CameraId"] == cid][["FrameId", "Id", "X", "Y", "Width", "Height"]].copy()
        mfid = int(gtd["FrameId"].max())
        gtd["FrameId"] += max_frame
        gtd = gtd.set_index(["FrameId", "Id"])
        gtds.append(gtd)

        if cid in ts_cams:
            tsd = pred[pred["CameraId"] == cid][["FrameId", "Id", "X", "Y", "Width", "Height"]].copy()
            mfid = max(mfid, int(tsd["FrameId"].max()))
            tsd["FrameId"] += max_frame
            tsd = tsd.set_index(["FrameId", "Id"])
            tsds.append(tsd)

        max_frame += mfid

    if not tsds:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0}

    acc = mm.utils.compare_to_groundtruth(pd.concat(gtds), pd.concat(tsds), "iou")
    mh = mm.metrics.create()
    metrics = ["idf1", "idp", "idr"]
    summary = mh.compute(acc, metrics=metrics, name="MultiCam")

    return {m: float(summary.loc["MultiCam", m]) for m in metrics}


# ---------------------------------------------------------------------------
# High-level helper: print results
# ---------------------------------------------------------------------------

def print_aic_results(results: Dict, label: str = "AIC"):
    """Print IDF1/IDP/IDR to stdout."""
    print(f"\n  [{label}]  IDF1={results['idf1']*100:.2f}%"
          f"  IDP={results['idp']*100:.2f}%"
          f"  IDR={results['idr']*100:.2f}%")
