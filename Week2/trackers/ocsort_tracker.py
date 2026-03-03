import numpy as np
from typing import List, Tuple
from .ocsort_tracker_original.ocsort import OCSort


class OCSORTTracker:
    """
    Wrapper for OC-SORT (Observation-Centric SORT) tracker.
    
    OC-SORT improves upon SORT by:
    - Using observation-centric momentum for better handling of occlusions
    - Implementing observation-centric re-association (OCR)
    - Supporting multiple association cost functions (IoU, GIoU, DIoU, etc.)
    """
    
    def __init__(
        self,
        det_thresh: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False
    ):
        """
        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is confirmed
            iou_threshold: Minimum IoU for matching
            delta_t: Time steps for observation-centric momentum
            asso_func: Association function ("iou", "giou", "ciou", "diou", "ct_dist")
            inertia: Weight for observation-centric momentum
            use_byte: Whether to use ByteTrack-style second matching with low-confidence detections
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.asso_func = asso_func
        self.inertia = inertia
        self.use_byte = use_byte
        
        self.tracker = OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte
        )
        self.tracks = {}  # {frame_id: [(track_id, bbox), ...]}
        
    def reset(self):
        """Reset the tracker to initial state."""
        self.tracker = OCSort(
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            delta_t=self.delta_t,
            asso_func=self.asso_func,
            inertia=self.inertia,
            use_byte=self.use_byte
        )
        self.tracks = {}
        
    def track(self, detections: List[Tuple], frame_id: int) -> List[Tuple]:
        """
        Track objects based on detections.
        
        Args:
            detections: List of detections in format [(x, y, w, h, conf), ...] or [(x, y, w, h), ...]
            frame_id: Current frame ID
            
        Returns:
            List of tracked objects in format [(track_id, (x, y, w, h)), ...]
        """
        if len(detections) == 0:
            # OCSORT expects update to be called even with empty detections
            output = self.tracker.update(np.empty((0, 5)), img_info=(1080, 1920), img_size=(1080, 1920))
            self.tracks[frame_id] = []
            return []
        
        # Convert detections from xywh to xyxy format with confidence
        xyxy_boxes = []
        
        for det in detections:
            if len(det) == 4:
                x, y, w, h = det
                conf = 1.0
            else:
                x, y, w, h, conf = det[:5]
            
            # Convert xywh to xyxy
            x1, y1, x2, y2 = x, y, x + w, y + h
            xyxy_boxes.append([x1, y1, x2, y2, conf])
        
        # Convert to numpy array
        detections_np = np.array(xyxy_boxes, dtype=np.float32)
        
        # Assume image size (can be adjusted if needed)
        # For AICity dataset, typical size is 1920x1080
        img_info = (1080, 1920)
        img_size = (1080, 1920)
        
        # Update tracker
        tracked_output = self.tracker.update(detections_np, img_info, img_size)
        
        # Convert tracked output from xyxy to xywh format
        tracked_objects = []
        
        if len(tracked_output) > 0:
            for track in tracked_output:
                x1, y1, x2, y2, track_id = track[:5]
                
                # Convert xyxy to xywh
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                tracked_objects.append((int(track_id), (x, y, w, h)))
        
        # Store tracking results
        self.tracks[frame_id] = tracked_objects
        
        return tracked_objects
