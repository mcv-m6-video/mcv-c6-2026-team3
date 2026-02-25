import cv2 as cv
from typing import Tuple, Callable
import os
import gc
from pathlib import Path
import numpy as np
from utils import *
import torch
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
import contextlib

class DetectionPipepline():
    """
    Complete detection pipeline to avoid repeating code.
    """
    
    def __init__(self, model, detector, preprocess_fn : Callable[[np.ndarray], np.ndarray] = None):
        self.model = model
        self.detector = detector
        self.preprocess_fn = preprocess_fn
        
    def _create_evaluator(self, annotations : Path, frame_size : Tuple[int, int], initial_frame : int) -> Tuple[COCOEvaluator, dict]:
        
        gt_data = get_COCO_gt(annotations, frame_size, initial_frame)
        def get_dataset():
            return gt_data

        if "video_dataset" not in DatasetCatalog:
            DatasetCatalog.register("video_dataset", get_dataset)
        
        MetadataCatalog.get("video_dataset").set(thing_classes=["object"])

        evaluator = COCOEvaluator("video_dataset", output_dir="./results/COCO_output")
        evaluator.reset()

        gt_dict = {d["image_id"]: d for d in gt_data}
        
        return evaluator, gt_dict
     
    def __call__(self, input : Path, output : Path, annotations : Path, bg_percentage : float, save : bool = True) -> float:
        
        cap = cv.VideoCapture(input)

        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

        frame_size = (width, height)
        
        if save:

            mask_out = cv.VideoWriter(os.path.join(output, "mask.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    cap.get(cv.CAP_PROP_FPS), 
                                    frame_size,
                                    isColor=False
                                    )
            bbox_out = cv.VideoWriter(os.path.join(output, "detections.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    cap.get(cv.CAP_PROP_FPS), 
                                    frame_size,
                                    isColor=True)
        
        total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        bg_frame_num = int(bg_percentage * total_frame_num)
        processed_frames = 0
        bg_frames = []

        evaluator, gt_dict = self._create_evaluator(annotations, frame_size, bg_frame_num)
        

        while True:

            ret, frame = cap.read()
            
            if not ret:
                break


            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if processed_frames < bg_frame_num:
                processed_frames += 1
                bg_frames.append(grey_frame)
                if processed_frames == bg_frame_num:
                    self.model.modelize_back(np.array(bg_frames))
                    del bg_frames
                    gc.collect() # Force garbage collection after releasing the heavy background buffer
                    
                continue
            
            mask = self.model(grey_frame).astype(np.uint8) * 255
        
            bboxes, detection = self.detector.detect(mask, processed_frames, self.preprocess_fn)

            instances = Instances((height, width))
            instances.pred_boxes = Boxes(bboxes)
            instances.scores = torch.ones(len(bboxes)) * 0.99
            instances.pred_classes = torch.zeros(len(bboxes))
            
            prediction = {
                "image_id" : processed_frames,
                "instances" : instances
            }
            
            evaluator.process([gt_dict[processed_frames]], [prediction])   

            if save: 
                mask_out.write(detection)
            
                # Draw ground truth boxes in red
                if processed_frames in gt_dict:
                    for gt_ann in gt_dict[processed_frames]["annotations"]:
                        x, y, w, h = gt_ann["bbox"]
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw predicted boxes in green
                for x1, y1, x2, y2 in bboxes:
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                bbox_out.write(frame)
            
            processed_frames += 1

        cap.release()
        
        print("Pipeline ended")
        
        if save:
            mask_out.release()
            bbox_out.release()


            with open(f"{output}/metrics.txt", "w") as f, contextlib.redirect_stdout(f):
                results = evaluator.evaluate()

            with open(f"{output}/metrics.txt", "a") as f:
                f.write("\n")
                f.write("--------------------------------------------------------------------------------\n")
                f.write("\n")
                f.write(f"mAP@05 : {results['bbox']['AP50']}")


            save_detections_txt(self.detector.detections, os.path.join(output, "detections.txt"))
            
            print(f"Results can be found inside {output} folder")
        
            return results['bbox']['AP50']
        
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                results = evaluator.evaluate()
        return results['bbox']['AP50']
