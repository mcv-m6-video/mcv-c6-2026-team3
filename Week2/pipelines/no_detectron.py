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

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_mean_iou(pred_boxes: List, gt_boxes: List, iou_threshold: float = 0.5) -> float:
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    gt_boxes_xyxy = [(x, y, x2, y2) for x, y, x2, y2 in gt_boxes]
    
    ious = []
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        for gt_box in gt_boxes_xyxy:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        
        if best_iou >= iou_threshold:
            ious.append(best_iou)
    
    if len(ious) == 0:
        return 0.0
    
    return sum(ious) / len(ious)
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
            
            if len(good_boxes) > 0 and len(gt[0]["boxes"]) > 0:
                frame_iou = compute_mean_iou(bboxes, gt[0]["boxes"], iou_threshold=0.5)
                all_ious.append(frame_iou)
                
            if save:
                
                for bbox in gt[0]["boxes"]:
                    x, y, x2, y2 = [p.item() for p in bbox]
                    cv.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
            
                for (x1, y1, x2, y2), score in zip(bboxes, scores):
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, f"{score:.2f}", (x1, y1 - 5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                bbox_out.write(frame)
                
        mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
            
        results = metric.compute()
        results['miou'] = mean_iou
            
        if save:
            bbox_out.release()
            with open(f"{output}/metrics.txt", "a") as f:
                f.write("\n")
                f.write("="*80 + "\n")
                f.write("SUMMARY METRICS\n")
                f.write("="*80 + "\n")
                f.write(f"mAP@0.5  : {results['map_50']:.4f}\n")
                f.write(f"mAP@0.75 : {results['map_75']:.4f}\n")
                f.write(f"mAP      : {results['map']:.4f}\n")
                f.write(f"mIoU     : {mean_iou:.4f}\n")
            
            save_detections_txt(self.detector.detections, os.path.join(output, "detections.txt"))
            
        return results
            
            