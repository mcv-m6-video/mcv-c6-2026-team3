import cv2 as cv
import numpy as np


class CCDetector:
    def __init__(self, min_pixels=50):
        self.min_pixels = min_pixels
        self.detections = {}  # {frame_id: [(x, y, w, h), ...]}
    
    def detect(self, mask, frame_id):
        binary_mask = (mask > 0).astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        bboxes = []
        
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            
            if area >= self.min_pixels:
                x = stats[i, cv.CC_STAT_LEFT]
                y = stats[i, cv.CC_STAT_TOP]
                w = stats[i, cv.CC_STAT_WIDTH]
                h = stats[i, cv.CC_STAT_HEIGHT]
                bboxes.append((x, y, w, h))
        
        self.detections[frame_id] = bboxes
        return bboxes
