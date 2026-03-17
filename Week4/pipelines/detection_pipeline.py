import cv2 as cv
from typing import Tuple, Dict, List
import os
from pathlib import Path
import numpy as np
from utils import *
import torch
import contextlib
from tqdm import tqdm


def compute_mean_iou(pred_boxes: List, gt_boxes: List, iou_threshold: float = 0.5) -> float:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    ious = []
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            ious.append(best_iou)
    
    if len(ious) == 0:
        print("Warning: No predictions matched with GT boxes above the IoU threshold.")
        return None
    
    return sum(ious) / len(ious)


class DetectionPipeline():
    def __init__(self, detector):
        self.detector = detector
        
     
    def __call__(self, input : Path, output : Path) -> Dict[str, float]:
        cap = cv.VideoCapture(str(input))

    
        
        total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        frame_id = 0
        
        print(f"Total frames: {total_frame_num}")
        

        with tqdm(total=total_frame_num, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                
                self.detector.detect(frame, frame_id)

                
                frame_id += 1
                pbar.update(1)

        cap.release()
                
        print("Pipeline ended")
        
        save_detections_txt(self.detector.detections, os.path.join(output, "detections.txt"))
        
        print(f"Results can be found inside {output} folder")
        
        
