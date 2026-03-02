import numpy as np
from typing import List, Tuple, Dict
from utils import compute_iou


class IOUTracker:
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}  # {frame_id: [(track_id, bbox), ...]}
        self.active_tracks = {}  # {track_id: bbox_xyxy}
        
    def xywh_to_xyxy(self, bbox):
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    
    def xyxy_to_xywh(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1, y1, x2 - x1, y2 - y1)
    
    def track(self, detections: List[Tuple], frame_id: int) -> List[Tuple]:
        tracked_objects = []
        
        # Convert detections to XYXY format for IoU computation
        detections_xyxy = [self.xywh_to_xyxy(det) for det in detections]
        
        if not self.active_tracks:
            # First frame or no active tracks: assign new IDs to all detections
            for det_xywh, det_xyxy in zip(detections, detections_xyxy):
                track_id = self.next_id
                self.next_id += 1
                self.active_tracks[track_id] = det_xyxy
                tracked_objects.append((track_id, det_xywh))
        else:
            # Match detections to existing tracks based on IoU
            matched_tracks = set()
            matched_detections = set()
            
            # Create IoU matrix
            track_ids = list(self.active_tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections_xyxy)))
            
            for i, track_id in enumerate(track_ids):
                track_bbox = self.active_tracks[track_id]
                for j, det_bbox in enumerate(detections_xyxy):
                    iou_matrix[i, j] = compute_iou(track_bbox, det_bbox)
            
            # Greedy matching: assign each detection to track with highest IoU
            while True:
                if iou_matrix.size == 0:
                    break
                
                # Find maximum IoU
                max_iou = np.max(iou_matrix)
                
                if max_iou < self.iou_threshold:
                    break
                
                # Get indices of maximum IoU
                track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                track_id = track_ids[track_idx]
                
                # Assign track ID to detection
                matched_tracks.add(track_id)
                matched_detections.add(det_idx)
                
                # Update active track with new bbox
                self.active_tracks[track_id] = detections_xyxy[det_idx]
                tracked_objects.append((track_id, detections[det_idx]))
                
                # Remove matched track and detection from matrix
                iou_matrix[track_idx, :] = -1
                iou_matrix[:, det_idx] = -1
            
            # Create new tracks for unmatched detections
            for j, (det_xywh, det_xyxy) in enumerate(zip(detections, detections_xyxy)):
                if j not in matched_detections:
                    track_id = self.next_id
                    self.next_id += 1
                    self.active_tracks[track_id] = det_xyxy
                    tracked_objects.append((track_id, det_xywh))
            
            # Remove unmatched tracks (lost objects)
            for track_id in track_ids:
                if track_id not in matched_tracks:
                    del self.active_tracks[track_id]
        
        # Store tracking results
        self.tracks[frame_id] = tracked_objects
        
        return tracked_objects
    
    def reset(self):
        self.next_id = 1
        self.tracks = {}
        self.active_tracks = {}
