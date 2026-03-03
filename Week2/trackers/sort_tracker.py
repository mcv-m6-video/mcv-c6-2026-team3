import numpy as np
from typing import List, Tuple, Dict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from utils import compute_iou


class KalmanBoxTracker:
    count = 0
    
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x_center
            [0, 1, 0, 0, 0, 1, 0],  # y_center
            [0, 0, 1, 0, 0, 0, 1],  # scale
            [0, 0, 0, 1, 0, 0, 0],  # aspect_ratio
            [0, 0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 0, 1]   # vs
        ])
        
        # Measurement function (we observe x, y, scale, aspect_ratio)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.0
        
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = self.bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z(bbox))
        
    def predict(self):
        # Prevent scale from becoming negative
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self.z_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        return self.z_to_bbox(self.kf.x)
        
    @staticmethod
    def bbox_to_z(bbox):
        x, y, w, h = bbox[:4]
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        scale = w * h
        aspect_ratio = w / float(h) if h != 0 else 1.0
        return np.array([x_center, y_center, scale, aspect_ratio]).reshape((4, 1))
        
    @staticmethod
    def z_to_bbox(x):
        x_center = x[0]
        y_center = x[1]
        scale = x[2]
        aspect_ratio = x[3]
        
        # Compute width and height from scale and aspect ratio
        w = np.sqrt(scale * aspect_ratio)
        h = scale / w if w != 0 else 1.0
        
        # Convert center to top-left corner
        x = x_center - w / 2.0
        y = y_center - h / 2.0
        
        return np.array([x, y, w, h]).flatten()


class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking
    Uses Kalman filtering for motion prediction and Hungarian algorithm for data association.
    """
    
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        max_age: Maximum number of frames to keep alive a track without associated detections
        min_hits: Minimum number of associated detections before track is confirmed
        iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.tracks = {}  # {frame_id: [(track_id, bbox), ...]}
        
    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.tracks = {}
        KalmanBoxTracker.count = 0
        
    def track(self, detections: List[Tuple], frame_id: int) -> List[Tuple]:
        self.frame_count += 1
        
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        dets = np.array([det[:4] for det in detections])
        
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
        for m in matched:
            self.trackers[m[0]].update(dets[m[1], :])
            
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        tracked_objects = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            
            d = trk.get_state()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                x, y, w, h = d
                tracked_objects.append((trk.id + 1, (int(x), int(y), int(w), int(h))))
                
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        self.tracks[frame_id] = tracked_objects
        
        return tracked_objects
        
    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
            
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))
            
        # Compute IoU matrix
        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        
        for t, trk in enumerate(trackers):
            trk_xyxy = (trk[0], trk[1], trk[0] + trk[2], trk[1] + trk[3])
            
            for d, det in enumerate(detections):
                det_xyxy = (det[0], det[1], det[0] + det[2], det[1] + det[3])
                iou_matrix[t, d] = compute_iou(trk_xyxy, det_xyxy)
                
        # Use Hungarian algorithm to solve the assignment problem
        # We want to maximize IoU, so we use (1 - IoU) as cost
        cost_matrix = 1 - iou_matrix
        
        # linear_sum_assignment finds minimum cost assignment
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
            
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 1]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if len(matches) == 0 or t not in matches[:, 0]:
                unmatched_trackers.append(t)
                
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
