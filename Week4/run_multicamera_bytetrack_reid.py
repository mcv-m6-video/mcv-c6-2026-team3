import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from trackers.bytetrack_tracker import ByteTrackTracker
from evaluation.tracking_eval import evaluate_tracking, load_gt_tracks

SEQUENCES = ["S01", "S03", "S04"]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-camera ByteTrack + Siamese ReID + IDF1")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--detections_root", default="outputs")
    parser.add_argument("--det_file", default="det_yolo.txt")
    parser.add_argument("--siamese_ckpt", required=True)
    parser.add_argument("--output_dir", default="outputs/tracking_bytetrack_reid")
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES)

    parser.add_argument("--track_thresh", type=float, default=0.25)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)

    parser.add_argument("--use_reid", action="store_true")
    parser.add_argument("--use_mtmc", action="store_true")
    parser.add_argument("--intra_cam_thr", type=float, default=0.85)
    parser.add_argument("--inter_cam_thr", type=float, default=0.65)
    parser.add_argument("--reid_max_gap", type=int, default=100)
    parser.add_argument("--min_move_pixels", type=float, default=50.0,
                        help="Min pixels a track must move to not be considered parked")

    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_siamese_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    resnet_base = models.resnet18(weights=None)
    extractor = nn.Sequential(*(list(resnet_base.children())[:-1]))
    model = nn.Sequential(
        extractor,
        nn.Flatten(),
        nn.BatchNorm1d(512),
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_detections(det_file: Path) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    dets = defaultdict(list)
    if not det_file.exists():
        return dets

    with open(det_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            x, y, w, h, conf = map(float, parts[2:7])
            dets[frame_id].append((x, y, w, h, conf))
    return dets


def build_crop_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def clip_box_xywh(box, width, height):
    x, y, w, h = box
    x1, y1 = max(0, min(int(x), width - 1)), max(0, min(int(y), height - 1))
    x2, y2 = max(0, min(int(x + w), width)), max(0, min(int(y + h), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_embeddings_for_tracks(frame, tracked_objects, model, device, transform):
    h, w = frame.shape[:2]
    crops = []
    valid_idx = []

    for i, (_, box) in enumerate(tracked_objects):
        clipped = clip_box_xywh(box, w, h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(transform(crop_rgb))
        valid_idx.append(i)

    emb_by_index = {}
    if not crops:
        return emb_by_index

    batch = torch.stack(crops, dim=0).to(device)
    with torch.no_grad():
        emb = model(batch)
        emb = nn.functional.normalize(emb, p=2, dim=1)

    emb_np = emb.detach().cpu().numpy()
    for local_i, track_i in enumerate(valid_idx):
        emb_by_index[track_i] = emb_np[local_i]
    return emb_by_index


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def merge_tracklets(tracklets: Dict[int, Dict], sim_thr: float, max_gap: int) -> Dict[int, int]:
    # sort tracklets chronologically by start frame to simulate real-time processing
    ordered_ids = sorted(tracklets.keys(), key=lambda tid: tracklets[tid]["start_frame"])
    
    global_tracks = [] # stores active global identities
    mapping = {}       # maps local track id to global track id
    next_gid = 1

    for tid in ordered_ids:
        t = tracklets[tid]
        best_gid = None
        best_sim = -1.0

        # compare current tracklet against existing global tracks
        for g in global_tracks:
            # temporal filter: prevent physical overlap and enforce max memory gap
            if t["start_frame"] <= g["end_frame"] or (t["start_frame"] - g["end_frame"]) > max_gap:
                continue
            
            # appearance matching via cosine similarity
            sim = cosine_sim(t["mean_emb"], g["mean_emb"])
            if sim > best_sim:
                best_sim, best_gid = sim, g["gid"]

        # if a valid match is found above the threshold
        if best_gid is not None and best_sim >= sim_thr:
            mapping[tid] = best_gid
            
            # update the global track's state
            for g in global_tracks:
                if g["gid"] == best_gid:
                    total_count = g["emb_count"] + t["emb_count"]
                    if total_count > 0:
                        # update embedding using a weighted moving average
                        g["mean_emb"] = (g["mean_emb"] * g["emb_count"] + t["mean_emb"] * t["emb_count"]) / total_count
                    g["emb_count"] = total_count
                    # extend the track's lifespan
                    g["end_frame"] = max(g["end_frame"], t["end_frame"])
                    break
        else:
            # no match found: initialize a new global identity
            mapping[tid] = next_gid
            global_tracks.append({
                "gid": next_gid, 
                "mean_emb": t["mean_emb"].copy(),
                "emb_count": t["emb_count"], 
                "end_frame": t["end_frame"],
            })
            next_gid += 1

    return mapping


def run_camera(video_path: Path, det_file: Path, gt_file: Path, model: nn.Module, transform, device: torch.device, args):
    # load offline yolo detections and init motion tracker
    detections = load_detections(det_file)
    tracker = ByteTrackTracker(
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, frame_rate=30,
    )

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records = []
    tracklets = {}

    # 1. frame-by-frame processing loop
    for frame_id in tqdm(range(1, total_frames + 1), desc=video_path.parent.name, leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # short-term association using bytetrack
        frame_dets = detections.get(frame_id, [])
        tracked_objects = tracker.track(frame_dets, frame_id)

        # extract appearance features (512d embeddings) for active tracks
        emb_by_index = {}
        if args.use_reid and tracked_objects:
            emb_by_index = extract_embeddings_for_tracks(frame, tracked_objects, model, device, transform)

        for i, (tid, box) in enumerate(tracked_objects):
            x, y, w, h = box
            records.append((frame_id, int(tid), int(x), int(y), int(w), int(h)))

            if not args.use_reid:
                continue

            # initialize tracklet memory if new identity
            if tid not in tracklets:
                tracklets[tid] = {
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "emb_sum": np.zeros(512, dtype=np.float32),
                    "emb_count": 0
                }
            tracklets[tid]["end_frame"] = frame_id

            # accumulate embeddings for robust mean representation
            if i in emb_by_index:
                tracklets[tid]["emb_sum"] += emb_by_index[i]
                tracklets[tid]["emb_count"] += 1
    cap.release()

    # 2. spatial filter: remove parked vehicles or background traffic
    track_centers = defaultdict(list)
    for frame_id, tid, x, y, w, h in records:
        cx, cy = x + w / 2.0, y + h / 2.0
        track_centers[tid].append((cx, cy))

    valid_moving_tids = set()
    for tid, centers in track_centers.items():
        if len(centers) < 2:
            continue
        
        # calculate euclidean distance between first and last observation
        first_center = centers[0]
        last_center = centers[-1]
        move_dist = ((last_center[0] - first_center[0])**2 + (last_center[1] - first_center[1])**2)**0.5

        if move_dist >= args.min_move_pixels:
            valid_moving_tids.add(tid)

    # purge static identities from memory
    records = [r for r in records if r[1] in valid_moving_tids]
    if args.use_reid:
        tracklets = {tid: t for tid, t in tracklets.items() if tid in valid_moving_tids}

    # 3. intra-camera tracklet stitching (handle long occlusions)
    local_tracklets = {}
    if args.use_reid:
        # compute mean embeddings before passing to the stitching algorithm
        cleaned_tracklets = {
            tid: {
                "start_frame": t["start_frame"],
                "end_frame": t["end_frame"],
                "emb_count": t["emb_count"],
                "mean_emb": t["emb_sum"] / t["emb_count"]
            }
            for tid, t in tracklets.items() if t["emb_count"] > 0
        }
        
        # map fragmented local ids to unified local ids
        id_map = merge_tracklets(cleaned_tracklets, args.intra_cam_thr, args.reid_max_gap)

        # recalculate global embeddings for the merged identities
        for old_tid, t in cleaned_tracklets.items():
            local_tid = id_map.get(old_tid, old_tid)
            if local_tid not in local_tracklets:
                local_tracklets[local_tid] = {
                    "emb_sum": t["mean_emb"] * t["emb_count"],
                    "emb_count": t["emb_count"]
                }
            else:
                local_tracklets[local_tid]["emb_sum"] += t["mean_emb"] * t["emb_count"]
                local_tracklets[local_tid]["emb_count"] += t["emb_count"]

        # finalize mean embeddings for the mtmc stage
        for local_tid, t in local_tracklets.items():
            t["mean_emb"] = t["emb_sum"] / max(t["emb_count"], 1)
    else:
        # fallback if reid is disabled: sequential mapping
        id_map = {tid: i + 1 for i, tid in enumerate(sorted({r[1] for r in records}))}

    # 4. output formatting and evaluation
    pred_tracks = defaultdict(list)
    remapped_records = []

    # apply the stitched id mapping to the raw bounding box records
    for frame_id, tid, x, y, w, h in records:
        new_tid = id_map.get(tid, tid)
        remapped_records.append((frame_id, new_tid, x, y, w, h))
        pred_tracks[frame_id].append((new_tid, (x, y, w, h)))

    # compute standard tracking metrics (idf1, hota, mota)
    gt_tracks = load_gt_tracks(str(gt_file), train_frames=0) if gt_file.exists() else {}
    metrics = evaluate_tracking(pred_tracks, gt_tracks) if gt_file.exists() else None
    
    return remapped_records, pred_tracks, gt_tracks, metrics, local_tracklets


def assign_sequence_global_ids(sequence_tracklets: Dict[Tuple[str, int], Dict], sim_thr: float) -> Dict[Tuple[str, int], int]:
    # sort keys to ensure deterministic processing order
    keys = sorted(sequence_tracklets.keys(), key=lambda x: (x[0], x[1]))
    if not keys:
        return {}

    # init disjoint-set structures
    parent = {k: k for k in keys}
    cams_in_component = {k: {k[0]} for k in keys}

    # find root of the set with path compression
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # merge two sets if they don't violate spatial constraints
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # strict anti-collision: prevent assigning same global id within the same camera
        if cams_in_component[ra] & cams_in_component[rb]:
            return False
        parent[rb] = ra
        cams_in_component[ra].update(cams_in_component[rb])
        return True

    # compute pairwise cosine similarity for all cross-camera tracklet pairs
    candidates = [(cosine_sim(sequence_tracklets[ka]["mean_emb"], sequence_tracklets[kb]["mean_emb"]), ka, kb)
                  for i, ka in enumerate(keys) for kb in keys[i + 1:] if ka[0] != kb[0]]
    
    # greedy matching: process highest confidence pairs first
    for sim, ka, kb in sorted([c for c in candidates if c[0] >= sim_thr], key=lambda x: x[0], reverse=True):
        union(ka, kb)

    # map graph components to sequential global ids
    root_to_gid = {}
    gid_map = {}
    next_gid = 1
    
    for key in keys:
        root = find(key)
        if root not in root_to_gid:
            root_to_gid[root] = next_gid
            next_gid += 1
        gid_map[key] = root_to_gid[root]

    return gid_map

def aggregate_global_eval(per_camera_data):
    pooled_pred = {}
    pooled_gt = {}
    frame_offset = 0

    for item in per_camera_data:
        pt, gt = item["pred_tracks"], item["gt_tracks"]
        max_frame = max([max(pt.keys()) if pt else 0, max(gt.keys()) if gt else 0])

        for frame_id, dets in pt.items():
            pooled_pred[frame_id + frame_offset] = [(tid, box) for tid, box in dets]
        if gt:
            for frame_id, gt_dict in gt.items():
                pooled_gt[frame_id + frame_offset] = dict(gt_dict)
        frame_offset += max_frame + 1

    return evaluate_tracking(pooled_pred, pooled_gt) if pooled_gt else None


def main():
    # parse arguments and initialize paths and device
    args = parse_args()
    data_root = Path(args.data_root)
    detections_root = Path(args.detections_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # load reid model and image transformations
    model = build_siamese_model(Path(args.siamese_ckpt), device)
    transform = build_crop_transform()

    summary_lines = ["sequence,camera,idf1,hota\n"]
    per_camera_data = []

    # iterate over selected challenge sequences
    for seq in args.sequences:
        seq_path = data_root / seq
        cam_dirs = sorted([p for p in seq_path.iterdir() if p.is_dir()])
        seq_results = []
        sequence_tracklets = {}
        seq_data_for_eval = []

        # --- phase 1: local tracking (mtsc) ---
        for cam_dir in cam_dirs:
            video_path = cam_dir / "vdo.avi"
            gt_file = cam_dir / "gt" / "gt.txt"
            det_file = detections_root / seq / cam_dir.name / args.det_file
            if not video_path.exists() or not det_file.exists():
                continue
            
            # execute bytetrack and feature extraction per camera
            remapped_records, pred_tracks, gt_tracks, metrics, local_tracklets = run_camera(
                video_path, det_file, gt_file, model, transform, device, args
            )

            # log local performance metrics
            if metrics:
                print(f"[{seq} - {cam_dir.name}] IDF1: {metrics['IDF1']:.4f}  HOTA: {metrics['HOTA']:.4f}")
                summary_lines.append(f"{seq},{cam_dir.name},{metrics['IDF1']:.6f},{metrics['HOTA']:.6f}\n")

            # store local results and tracklet embeddings for global association
            seq_results.append({
                "seq": seq,
                "cam": cam_dir.name,
                "records": remapped_records,
                "gt_tracks": gt_tracks,
                "local_ids": sorted({r[1] for r in remapped_records})
            })
            for local_tid, t in local_tracklets.items():
                sequence_tracklets[(cam_dir.name, local_tid)] = t

        # --- phase 2: global association (mtmc) ---
        global_id_map = {}
        if args.use_mtmc and args.use_reid and sequence_tracklets:
            # apply spatial-temporal graph matching across all cameras
            global_id_map = assign_sequence_global_ids(sequence_tracklets, args.inter_cam_thr)
        
        # assign isolated tracklets a unique global id as fallback
        next_gid = max(global_id_map.values()) + 1 if global_id_map else 1
        for item in seq_results:
            for local_tid in item["local_ids"]:
                key = (item["cam"], local_tid)
                if key not in global_id_map:
                    global_id_map[key] = next_gid
                    next_gid += 1

        # group cameras by global id
        gid_to_cams = defaultdict(set)
        for (cam, local_tid), gid in global_id_map.items():
            gid_to_cams[gid].add(cam)

        # keep only vehicles seen in multiple cameras
        valid_gids = {gid for gid, cams in gid_to_cams.items() if len(cams) > 1}

        # --- phase 3: formatting and evaluation ---
        for item in seq_results:
            cam = item["cam"]
            pred_tracks_global = defaultdict(list)
            mot_lines = []

            for frame_id, local_tid, x, y, w, h in item["records"]:
                gid = global_id_map[(cam, local_tid)]
                if gid not in valid_gids:
                    continue
                pred_tracks_global[frame_id].append((gid, (x, y, w, h)))
                mot_lines.append(f"{frame_id},{gid},{x},{y},{w},{h},1,-1,-1,-1\n")

            # sort predictions chronologically
            mot_lines.sort(key=lambda line: (int(line.split(",")[0]), int(line.split(",")[1])))
            out_cam_dir = output_root / seq / cam
            out_cam_dir.mkdir(parents=True, exist_ok=True)

            # export final tracking results
            with open(out_cam_dir / "tracking_bytetrack_reid.txt", "w") as f:
                f.writelines(mot_lines)

            cam_data = {
                "seq": seq,
                "cam": cam,
                "pred_tracks": pred_tracks_global,
                "gt_tracks": item["gt_tracks"]
            }
            per_camera_data.append(cam_data)
            seq_data_for_eval.append(cam_data)
            
        # compute sequence-level multi-camera metrics
        seq_metrics = aggregate_global_eval(seq_data_for_eval)
        if seq_metrics:
            print(f"\n=> SEQUENCE {seq} MULTI-CAMERA: IDF1={seq_metrics['IDF1']:.4f}  HOTA={seq_metrics['HOTA']:.4f}\n")
            summary_lines.append(f"{seq},ALL,{seq_metrics['IDF1']:.6f},{seq_metrics['HOTA']:.6f}\n")

    # compute overall multi-camera metrics across all sequences
    global_metrics = aggregate_global_eval(per_camera_data)
    if global_metrics:
        print(f"\n=== GLOBAL MULTI-CAMERA ===\nIDF1={global_metrics['IDF1']:.4f}  HOTA={global_metrics['HOTA']:.4f}")
        summary_lines.append(f"GLOBAL,ALL,{global_metrics['IDF1']:.6f},{global_metrics['HOTA']:.6f}\n")

    # export summary metrics to csv
    with open(output_root / "summary_metrics.csv", "w") as f:
        f.writelines(summary_lines)

if __name__ == "__main__":
    main()