import xml.etree.ElementTree as ET
import numpy as np
import contextlib
import os
from typing import Dict
import trackeval

# numpy retrocompat
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_tracks(xml_path: str, train_frames: int = 0) -> Dict[int, Dict]:
    root = ET.parse(xml_path).getroot()
    gt_tracks = {}

    for track in root.findall('track'):
        if track.get('label') != 'car':
            continue
        tid = int(track.get('id'))
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            if frame < train_frames:
                continue
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            gt_tracks.setdefault(frame, {})[tid] = (xtl, ytl, xbr - xtl, ybr - ytl)

    return gt_tracks


def evaluate_tracking(pred_tracks: Dict, gt_tracks: Dict) -> Dict[str, float]:
    all_frames = sorted(set(pred_tracks.keys()) | set(gt_tracks.keys()))

    all_gt_ids = set(tid for fgt in gt_tracks.values() for tid in fgt)
    all_pred_ids = set(tid for fp in pred_tracks.values() for tid, _ in fp)

    gt_id_map = {v: i for i, v in enumerate(sorted(all_gt_ids))}
    pred_id_map = {v: i for i, v in enumerate(sorted(all_pred_ids))}

    gt_ids_list, tracker_ids_list = [], []
    gt_dets_list, tracker_dets_list = [], []
    similarity_scores_list = []

    for frame_id in all_frames:
        gt_frame = gt_tracks.get(frame_id, {})
        gt_ids = np.array([gt_id_map[t] for t in gt_frame], dtype=int)
        gt_bboxes = np.array([[x, y, x+w, y+h] for x, y, w, h in gt_frame.values()], dtype=float)
        if len(gt_ids) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=float)

        pred_dets = pred_tracks.get(frame_id, [])
        pred_ids = np.array([pred_id_map[t] for t, _ in pred_dets], dtype=int)
        pred_bboxes = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for _, b in pred_dets], dtype=float)
        if len(pred_ids) == 0:
            pred_bboxes = np.zeros((0, 4), dtype=float)

        iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=float)
        for i, gb in enumerate(gt_bboxes):
            for j, pb in enumerate(pred_bboxes):
                iou_matrix[i, j] = compute_iou(
                    (gb[0], gb[1], gb[2]-gb[0], gb[3]-gb[1]),
                    (pb[0], pb[1], pb[2]-pb[0], pb[3]-pb[1])
                )

        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(pred_ids)
        gt_dets_list.append(gt_bboxes)
        tracker_dets_list.append(pred_bboxes)
        similarity_scores_list.append(iou_matrix)

    data = {
        'num_timesteps': len(all_frames),
        'num_gt_ids': len(gt_id_map),
        'num_tracker_ids': len(pred_id_map),
        'num_gt_dets': sum(len(x) for x in gt_ids_list),
        'num_tracker_dets': sum(len(x) for x in tracker_ids_list),
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'gt_dets': gt_dets_list,
        'tracker_dets': tracker_dets_list,
        'similarity_scores': similarity_scores_list,
    }

    hota_metric = trackeval.metrics.HOTA({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})
    identity_metric = trackeval.metrics.Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        hota_res = hota_metric.eval_sequence(data)
        idf1_res = identity_metric.eval_sequence(data)

    return {'HOTA': float(np.mean(hota_res['HOTA'])), 'IDF1': idf1_res['IDF1']}
