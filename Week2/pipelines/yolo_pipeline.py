import cv2 as cv
from typing import Tuple, Dict, List
import os
from pathlib import Path
import numpy as np
from utils import *
import torch
from torch.utils.data import DataLoader
from detectron2.structures import Boxes, Instances
from datasets import AICityDataset
import contextlib
from tqdm import tqdm


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
    
    gt_boxes_xyxy = [(x, y, x + w, y + h) for x, y, w, h in gt_boxes]
    
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


class EvaluationPipeline():
    
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
                shuffle=False    
            )
        else:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False    
            )
        
        
        
        width = dataset.width
        height = dataset.height
        fps = dataset.fps
        frame_size = (width, height)
        
        
        if save:
            bbox_out = cv.VideoWriter(os.path.join(output, "detections.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    fps, 
                                    frame_size,
                                    isColor=True)
        
        # For mIoU calculation
        all_ious = []

        for frame_id, (frame, coco_gt) in tqdm(enumerate(data_loader, start=initial_id), desc="Processed frames: ", unit=" frames", total=len(data_loader)):
            
            frame = np.array(frame[0])
            
            fixed_gt = {
                "file_name": coco_gt["file_name"],
                "image_id": int(coco_gt["image_id"].item()),
                "height": int(coco_gt["height"].item()),
                "width": int(coco_gt["width"].item()),
                "annotations": [
                    {
                        "bbox": [int(x.item()) for x in ann["bbox"]],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(ann["category_id"].item())
                    }
                    for ann in coco_gt["annotations"]
                ]
            }
            
            bboxes, scores = self.detector.detect(frame, frame_id)

            instances = Instances((height, width))
            instances.pred_boxes = Boxes(torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)))
            instances.scores = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
            instances.pred_classes = torch.zeros(len(bboxes), dtype=torch.int64) if bboxes else torch.zeros(0, dtype=torch.int64)
            
            prediction = {
                "image_id" : frame_id,
                "instances" : instances
            }
            
            evaluator.process([fixed_gt], [prediction])
            
            gt_boxes = [ann["bbox"] for ann in fixed_gt["annotations"]]
            if len(bboxes) > 0 and len(gt_boxes) > 0:
                frame_iou = compute_mean_iou(bboxes, gt_boxes, iou_threshold=0.5)
                all_ious.append(frame_iou)  
                
            if save: 
                
                for gt_ann in fixed_gt["annotations"]:
                    x, y, x2, y2 = gt_ann["bbox"]
                    cv.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                
                for (x1, y1, x2, y2), score in zip(bboxes, scores):
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, f"{score:.2f}", (x1, y1 - 5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                bbox_out.write(frame)
        
        mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
                
        print("Pipeline ended")
        
        if save:
            bbox_out.release()

            with open(f"{output}/metrics.txt", "w") as f, contextlib.redirect_stdout(f):
                results = evaluator.evaluate()

            with open(f"{output}/metrics.txt", "a") as f:
                f.write("\n")
                f.write("="*80 + "\n")
                f.write("SUMMARY METRICS\n")
                f.write("="*80 + "\n")
                f.write(f"mAP@0.5  : {results['bbox']['AP50']:.4f}\n")
                f.write(f"mAP@0.75 : {results['bbox']['AP75']:.4f}\n")
                f.write(f"mAP      : {results['bbox']['AP']:.4f}\n")
                f.write(f"mIoU     : {mean_iou:.4f}\n")

            save_detections_txt(self.detector.detections, os.path.join(output, "detections.txt"))
            
            print(f"Results can be found inside {output} folder")
        
            return {
                'mAP50': results['bbox']['AP50'],
                'mAP75': results['bbox']['AP75'],
                'mAP': results['bbox']['AP'],
                'mIoU': mean_iou
            }
        
        # If not saving, just evaluate
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            results = evaluator.evaluate()
            
        return {
            'mAP50': results['bbox']['AP50'],
            'mAP75': results['bbox']['AP75'],
            'mAP': results['bbox']['AP'],
            'mIoU': mean_iou
        }
