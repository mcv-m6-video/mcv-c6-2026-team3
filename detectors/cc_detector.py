import cv2 as cv
import numpy as np
from typing import Tuple

class CCDetector:
    def __init__(self, min_pixels=50):
        self.min_pixels = min_pixels
        self.detections = {}  # {frame_id: [(x, y, w, h), ...]}


    def _preprocess(self, mask : np.ndarray) -> cv.Mat | np.ndarray:

        #With this we get rid of small noise arround the image. Cars and byciles
        #are supposed to be rather big, so we can go ahead with a big structuring
        #element
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        #Now we try to connect the resulting components along the image
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,10))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        #We compute boxes of each component and fill them to get rid
        #of smaller boxes inside good detections
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):
            
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            
            new_mask[y:y+h, x:x+w] = 255


        return new_mask
    
    def detect(self, mask : np.ndarray, frame_id : int, preprocess : bool = False) -> Tuple[list, cv.Mat | np.ndarray]:
        
        new_mask = mask
        
        if preprocess:
            new_mask = self._preprocess(mask)
       
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(new_mask, connectivity=8)
        
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
        return bboxes, new_mask
