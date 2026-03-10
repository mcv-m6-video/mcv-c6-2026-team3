import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1: return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

class FlowIOUTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.active_tracks = {}
        self.track_ages = {}

    def predict_with_flow(self, flow_u, flow_v):
        predicted_tracks = {}
        H, W = flow_u.shape
        
        for tid, bbox in self.active_tracks.items():
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)

            # median flow over inner 80% to avoid edge background
            bw, bh = x2 - x1, y2 - y1
            if bw > 1 and bh > 1:
                cx, cy = x1 + bw / 2, y1 + bh / 2
                inner_r = 0.8
                
                xx1 = int(max(0, cx - bw * inner_r / 2))
                yy1 = int(max(0, cy - bh * inner_r / 2))
                xx2 = int(min(W - 1, cx + bw * inner_r / 2))
                yy2 = int(min(H - 1, cy + bh * inner_r / 2))
                
                if xx2 > xx1 and yy2 > yy1:
                    med_u = np.median(flow_u[yy1:yy2, xx1:xx2])
                    med_v = np.median(flow_v[yy1:yy2, xx1:xx2])
                else:
                    med_u, med_v = 0, 0
            else:
                med_u, med_v = 0, 0
            
            predicted_tracks[tid] = [bbox[0] + med_u, bbox[1] + med_v, 
                                     bbox[2] + med_u, bbox[3] + med_v]
        return predicted_tracks

    def track(self, detections_xyxy, flow_u=None, flow_v=None):
        if flow_u is not None and self.active_tracks:
            projected = self.predict_with_flow(flow_u, flow_v)
        else:
            projected = self.active_tracks.copy()

        track_ids = list(projected.keys())
        if not track_ids:
            for det in detections_xyxy:
                self._add_new_track(det)
            return [(tid, self.active_tracks[tid]) for tid in self.active_tracks]

        iou_matrix = np.zeros((len(track_ids), len(detections_xyxy)))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections_xyxy):
                iou_matrix[i, j] = compute_iou(projected[tid], det)

        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
        
        matched_tracks = set()
        matched_dets = set()
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.iou_threshold:
                tid = track_ids[i]
                self.active_tracks[tid] = detections_xyxy[j]
                self.track_ages[tid] = 0
                matched_tracks.add(tid)
                matched_dets.add(j)

        for j, det in enumerate(detections_xyxy):
            if j not in matched_dets: self._add_new_track(det)
            
        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self.track_ages[tid] += 1
                if self.track_ages[tid] >= self.max_age:
                    del self.active_tracks[tid]
                    del self.track_ages[tid]
                else:
                    self.active_tracks[tid] = projected[tid]

        return [(tid, self.active_tracks[tid]) for tid in self.active_tracks if self.track_ages[tid] == 0]

    def _add_new_track(self, bbox):
        self.active_tracks[self.next_id] = bbox
        self.track_ages[self.next_id] = 0
        self.next_id += 1