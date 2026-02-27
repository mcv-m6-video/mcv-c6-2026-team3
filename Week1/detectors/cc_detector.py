import cv2 as cv
import numpy as np
from typing import Tuple, Callable

class CCDetector:
    def __init__(self, min_pixels=50):
        self.min_pixels = min_pixels
        self.detections = {}  # {frame_id: [(x, y, w, h), ...]}
    
    def detect(self, mask : np.ndarray, frame_id : int, preprocess : Callable[[np.ndarray], np.ndarray] = None) -> Tuple[list, cv.Mat | np.ndarray]:
        
        new_mask = mask
        
        if preprocess:
            new_mask = preprocess(mask)
       
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(new_mask, connectivity=8)
        
        bboxes = []
        
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            
            if area >= self.min_pixels:
                x1 = stats[i, cv.CC_STAT_LEFT]
                y1 = stats[i, cv.CC_STAT_TOP]
                x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
                y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]
                bboxes.append((x1, y1, x2, y2))
        
        self.detections[frame_id] = bboxes
        return bboxes, new_mask
