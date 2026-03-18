import numpy as np
from typing import List, Tuple

from supervision import ByteTrack, Detections


class ByteTrackTracker:
    """Simple ByteTrack wrapper returning (track_id, (x, y, w, h))."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracker = ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )
        self.tracks = {}

    def reset(self):
        self.tracker = ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
        )
        self.tracks = {}

    def track(self, detections: List[Tuple], frame_id: int) -> List[Tuple]:
        if len(detections) == 0:
            self.tracks[frame_id] = []
            return []

        xyxy_boxes = []
        confidences = []

        for det in detections:
            if len(det) == 4:
                x, y, w, h = det
                conf = 1.0
            else:
                x, y, w, h, conf = det[:5]

            x1, y1, x2, y2 = x, y, x + w, y + h
            xyxy_boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

        sv_detections = Detections(
            xyxy=np.array(xyxy_boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(xyxy_boxes), dtype=int),
        )

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        tracked_objects = []
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                x1, y1, x2, y2 = tracked_detections.xyxy[i]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                tracked_objects.append((int(track_id), (x, y, w, h)))

        self.tracks[frame_id] = tracked_objects
        return tracked_objects
