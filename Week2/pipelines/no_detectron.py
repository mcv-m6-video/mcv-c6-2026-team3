import cv2 as cv
from typing import Tuple, Dict, List
import os
from pathlib import Path
import numpy as np
from utils import *
import torch
from torch.utils.data import DataLoader
from detectron2.structures import Boxes, Instances
from datasets import AICityDataset, collate_fn
import contextlib
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class NoDetectronPipeline():
    
    def __init__(self, detector):
        self.detector = detector
     
    def __call__(
        self, 
        dataset : AICityDataset,
        output : str,
        subset = None,
        initial_id : int = 0,
        save : bool = True
    ) -> Dict[str, float]:
        
        if subset:
            data_loader = DataLoader(
                dataset=subset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn    
            )
        else:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn  
            )
        
        width = dataset.width
        height = dataset.height
        fps = dataset.fps
        frame_size = (width, height)
        
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
        metric.reset()
        
        if save:
            bbox_out = cv.VideoWriter(os.path.join(output, "detections.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    fps, 
                                    frame_size,
                                    isColor=True)
        
        # For mIoU calculation
        all_ious = []

        for frame_id, (frame, gt) in tqdm(enumerate(data_loader, start=initial_id), desc="Processed frames: ", unit=" frames", total=len(data_loader)):
            
            frame = np.array(frame[0])
            
            bboxes, scores = self.detector.detect(frame, frame_id)
            
            good_boxes = torch.tensor(bboxes, dtype=torch.float32)
            good_scores = torch.tensor(scores)
            good_labels = torch.zeros_like(good_scores, dtype=int)
            
            pred = {
                "boxes" : good_boxes,
                "scores" : good_scores,
                "labels" : good_labels
            }
            
            metric.update([pred], gt)
            
        metric = metric.compute()
        print(metric)
            
            