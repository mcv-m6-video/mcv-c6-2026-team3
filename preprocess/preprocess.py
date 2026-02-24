import cv2 as cv
import numpy as np
from typing import Tuple

def generate_morph_func(open_size : int, close_size : Tuple[int, int]):
    
    def pre_morph(mask : np.ndarray) -> np.ndarray:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (open_size,open_size))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        
        #Now we try to connect the resulting components along the image
        kernel = cv.getStructuringElement(cv.MORPH_RECT, close_size)
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
    
    return pre_morph

def preprocess_morph(mask : np.ndarray) -> np.ndarray:

    #With this we get rid of small noise arround the image. Cars and byciles
    #are supposed to be rather big, so we can go ahead with a big structuring
    #element
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,10))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    #Now we try to connect the resulting components along the image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30,30))
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