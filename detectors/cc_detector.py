import cv2 as cv
import numpy as np
from typing import Tuple, Callable

class CCDetector:
    def __init__(self, min_pixels=50):
        self.min_pixels = min_pixels
        self.detections = {}  # {frame_id: [(x, y, w, h), ...]}
    
    def detect(self, mask : np.ndarray, frame_id : int, preprocess : Callable[[np.ndarray], np.ndarray] = None,
               mask_morph=None, mask_bounding = None, normal_frame : np.ndarray = None) -> Tuple[list, cv.Mat | np.ndarray]:
        
        new_mask = mask
        
        if preprocess:
            new_mask = preprocess(mask, mask_morph)
       
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(new_mask, connectivity=8)
        
        bboxes = []
        
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            x1 = stats[i, cv.CC_STAT_LEFT]
            y1 = stats[i, cv.CC_STAT_TOP]
            x2 = x1 + stats[i, cv.CC_STAT_WIDTH]
            y2 = y1 + stats[i, cv.CC_STAT_HEIGHT]
            
            cv.rectangle(normal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if area >= self.min_pixels:
                bboxes.append((x1, y1, x2, y2))
        
        mask_bounding.write(normal_frame)

        self.detections[frame_id] = bboxes
        return bboxes, new_mask
