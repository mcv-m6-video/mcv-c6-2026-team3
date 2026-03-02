import cv2 as cv
from typing import Dict, Optional
import os
from pathlib import Path
import numpy as np
from utils import *
import contextlib
import trackeval

#Numpy retrocompatibility for older versions of trackeval
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool


def load_gt_tracks(xml_path: str, train_frames: int = 0) -> Dict[int, Dict]:
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    gt_tracks = {}
    
    for track in root.findall('track'):
        label = track.get('label')
        if label != 'car':
            continue
        
        track_id = int(track.get('id'))
        
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            
            if frame < train_frames:
                continue
            
            ytl = int(float(box.get('ytl')))
            xtl = int(float(box.get('xtl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            
            x, y = xtl, ytl
            w = xbr - xtl
            h = ybr - ytl
            
            if frame not in gt_tracks:
                gt_tracks[frame] = {}
            
            gt_tracks[frame][track_id] = (x, y, w, h)
    
    return gt_tracks


def evaluate_tracking(pred_tracks: Dict, gt_tracks: Dict) -> Dict[str, float]:
    all_frames = sorted(set(pred_tracks.keys()) | set(gt_tracks.keys()))
    
    # Create ID mappings to ensure sequential IDs starting from 0 (TrackEval requires this)
    all_gt_ids = set(tid for frame_gt in gt_tracks.values() for tid in frame_gt.keys())
    all_pred_ids = set(tid for frame_pred in pred_tracks.values() for tid, _ in frame_pred)
    
    gt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_gt_ids))}
    pred_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_pred_ids))}
    
    # Prepare data arrays
    gt_ids_list = []
    tracker_ids_list = []
    gt_dets_list = []
    tracker_dets_list = []
    similarity_scores_list = []
    
    for frame_id in all_frames:
        # Ground truth
        gt_track = gt_tracks.get(frame_id, {})
        gt_ids = np.array([gt_id_map[tid] for tid in gt_track.keys()], dtype=int)
        gt_bboxes = np.array([[x, y, x+w, y+h] for x, y, w, h in gt_track.values()], dtype=float)
        if len(gt_ids) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=float)
        
        # Predictions
        pred_dets = pred_tracks.get(frame_id, [])
        pred_ids = np.array([pred_id_map[tid] for tid, _ in pred_dets], dtype=int)
        pred_bboxes = np.array([[x, y, x+w, y+h] for _, (x, y, w, h) in pred_dets], dtype=float)
        if len(pred_ids) == 0:
            pred_bboxes = np.zeros((0, 4), dtype=float)
        
        # Compute IoU similarity matrix
        if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
            iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=float)
        else:
            iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=float)
            for i, gt_box in enumerate(gt_bboxes):
                for j, pred_box in enumerate(pred_bboxes):
                    iou_matrix[i, j] = compute_iou(
                        (gt_box[0], gt_box[1], gt_box[2], gt_box[3]),
                        (pred_box[0], pred_box[1], pred_box[2], pred_box[3])
                    )
        
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(pred_ids)
        gt_dets_list.append(gt_bboxes)
        tracker_dets_list.append(pred_bboxes)
        similarity_scores_list.append(iou_matrix)
    
    # Prepare data dict with all required fields
    data = {
        'num_timesteps': len(all_frames),
        'num_gt_ids': len(gt_id_map),
        'num_tracker_ids': len(pred_id_map),
        'num_gt_dets': sum(len(gt_ids) for gt_ids in gt_ids_list),
        'num_tracker_dets': sum(len(pred_ids) for pred_ids in tracker_ids_list),
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
    
    # Extract scalar values (TrackEval may return arrays)
    hota = float(np.mean(hota_res['HOTA']))
    idf1 = idf1_res['IDF1']

    return {'HOTA': hota, 'IDF1': idf1}


class TrackingPipeline:
    def __init__(self, tracker, detector=None):
        self.tracker = tracker
        self.detector = detector
        
    def __call__(self, 
                 input_path: Path, 
                 output_path: Path, 
                 annotations_path: Path,
                 train_percentage: float = 0.25,
                 detections_file: Optional[Path] = None,
                 save: bool = True) -> Dict[str, float]:
        self.tracker.reset()
        
        if detections_file:
            print(f"Loading detections from {detections_file}")
            detections = load_detections_txt(detections_file)
        else:
            if not self.detector:
                raise ValueError("Either detector or detections_file must be provided")
            print("Running detection...")
            detections = self._run_detection(input_path, train_percentage)
        
        # Open video
        cap = cv.VideoCapture(str(input_path))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_size = (width, height)
        total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        train_frame_num = int(train_percentage * total_frame_num)
        
        if save:
            track_out = cv.VideoWriter(os.path.join(output_path, "tracking.avi"),
                                      cv.VideoWriter_fourcc(*'XVID'),
                                      cap.get(cv.CAP_PROP_FPS),
                                      frame_size,
                                      isColor=True)
        
        print(f"Test frames (tracked): {train_frame_num}-{total_frame_num-1}")
        print(f"Running tracking...")
        
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip train frames
            if frame_id < train_frame_num:
                frame_id += 1
                continue
            
            # Get detections and track
            frame_detections = detections.get(frame_id, [])
            tracked_objects = self.tracker.track(frame_detections, frame_id)
            
            if save:
                # Draw tracking results
                for track_id, bbox in tracked_objects:
                    x, y, w, h = bbox
                    x2, y2 = x + w, y + h
                    
                    # Different colors for different track IDs
                    color = tuple([int(c) for c in np.random.RandomState(track_id).randint(0, 255, 3)])
                    
                    cv.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv.putText(frame, f"ID: {track_id}", (x, y - 5),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                track_out.write(frame)
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                print(f"Tracked {frame_id}/{total_frame_num} frames...")
        
        cap.release()
        if save:
            track_out.release()
        
        print("Tracking completed")

        print("Loading ground truth...")
        gt_tracks = load_gt_tracks(annotations_path, train_frame_num)
        print("Evaluating tracking...")
        metrics = evaluate_tracking(self.tracker.tracks, gt_tracks)
        
        if save:
            # Save tracking results
            tracking_file = output_path / "tracking.txt"
            save_tracking_txt(self.tracker.tracks, tracking_file)
            
            # Save metrics
            with open(output_path / "metrics.txt", "w") as f:
                f.write("="*80 + "\n")
                f.write("TRACKING METRICS\n")
                f.write("="*80 + "\n")
                f.write(f"HOTA: {metrics['HOTA']:.4f}\n")
                f.write(f"IDF1: {metrics['IDF1']:.4f}\n")
            
            print(f"Results saved to {output_path}")
            
        return metrics
    
    
    def _run_detection(self, input_path: Path, train_percentage: float) -> Dict:
        cap = cv.VideoCapture(str(input_path))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        train_frames = int(train_percentage * total_frames)
        
        detections = {}
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id < train_frames:
                frame_id += 1
                continue
            
            bboxes, scores = self.detector.detect(frame, frame_id)
            
            # Convert XYXY to XYWH
            frame_detections = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                frame_detections.append((x1, y1, w, h))
            
            detections[frame_id] = frame_detections
            frame_id += 1
            
            if frame_id % 100 == 0:
                print(f"Detected {frame_id}/{total_frames} frames...")
        
        cap.release()
        return detections
