import cv2 as cv
from typing import Tuple, Dict, List
import os
from pathlib import Path
import numpy as np
from utils import *
import torch
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
import contextlib


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


class DetectionPipeline():
    def __init__(self, detector):
        self.detector = detector
        
    def _create_evaluator(self, annotations : Path, frame_size : Tuple[int, int], initial_frame : int = 0) -> Tuple[COCOEvaluator, dict]:
        gt_data = get_COCO_gt(annotations, frame_size, initial_frame)
        
        def get_dataset():
            return gt_data

        if "video_dataset" not in DatasetCatalog:
            DatasetCatalog.register("video_dataset", get_dataset)
        
        MetadataCatalog.get("video_dataset").set(thing_classes=["car"])

        evaluator = COCOEvaluator("video_dataset", output_dir="./results/COCO_output")
        evaluator.reset()

        gt_dict = {d["image_id"]: d for d in gt_data}
        
        return evaluator, gt_dict
     
    def __call__(self, input : Path, output : Path, annotations : Path, train_percentage : float = 0.25, save : bool = True) -> Dict[str, float]:
        cap = cv.VideoCapture(str(input))

        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_size = (width, height)
        
        if save:
            bbox_out = cv.VideoWriter(os.path.join(output, "detections.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    cap.get(cv.CAP_PROP_FPS), 
                                    frame_size,
                                    isColor=True)
        
        total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        train_frame_num = int(train_percentage * total_frame_num)
        frame_id = 0

        evaluator, gt_dict = self._create_evaluator(annotations, frame_size, initial_frame=train_frame_num)
        
        # For mIoU calculation
        all_ious = []
        
        print(f"Total frames: {total_frame_num}")
        print(f"Train frames (skipped): 0-{train_frame_num-1}")
        print(f"Test frames (evaluated): {train_frame_num}-{total_frame_num-1}")
        print(f"Processing test frames...")

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip train frames
            if frame_id < train_frame_num:
                frame_id += 1
                continue
            
            bboxes, scores = self.detector.detect(frame, frame_id)

            instances = Instances((height, width))
            instances.pred_boxes = Boxes(torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)))
            instances.scores = torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
            instances.pred_classes = torch.zeros(len(bboxes), dtype=torch.int64) if bboxes else torch.zeros(0, dtype=torch.int64)
            
            prediction = {
                "image_id" : frame_id,
                "instances" : instances
            }
            
            if frame_id in gt_dict:
                evaluator.process([gt_dict[frame_id]], [prediction])   
                
                gt_boxes = [ann["bbox"] for ann in gt_dict[frame_id]["annotations"]]
                if len(bboxes) > 0 and len(gt_boxes) > 0:
                    frame_iou = compute_mean_iou(bboxes, gt_boxes, iou_threshold=0.5)
                    all_ious.append(frame_iou)   

            if save: 
                if frame_id in gt_dict:
                    for gt_ann in gt_dict[frame_id]["annotations"]:
                        x, y, w, h = gt_ann["bbox"]
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                for (x1, y1, x2, y2), score in zip(bboxes, scores):
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, f"{score:.2f}", (x1, y1 - 5), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                bbox_out.write(frame)
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{total_frame_num} frames...")

        cap.release()
        
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
