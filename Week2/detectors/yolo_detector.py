import cv2 as cv
import numpy as np
import os
from typing import Tuple
from ultralytics import YOLO
from pathlib import Path

class YOLODetector:
    def __init__(self, model_name : str = "yolo26n.pt", model = None, finetune = False):
        # Create models directory if it doesn't exist
        if not model:
            models_dir = Path(__file__).parent.parent / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Build model path
            model_path = models_dir / model_name
            
            # Load YOLO model (will download to models/ if not exists)
            self.model = YOLO(str(model_path))
        else:
            self.model = model
        self.car_class_id = 2  # Car class in COCO dataset
        self.finetuned = finetune
        self.truck_class_id = 7  # Truck class in COCO dataset
        self.detections = {}  # {frame_id: [(x, y, w, h, conf), ...]}
        self.scores = {}  # {frame_id: [score1, score2, ...]} - kept for backward compatibility
    
    def detect(self, frame : np.ndarray, frame_id : int) -> Tuple[list, list]:
        
        results = self.model.predict(
            frame,
            imgsz=640,
            verbose=False,
            end2end=False
        )
        
        bboxes = []
        scores = []
        detections_xywh = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                
                # Only keep car and truck detections
                if class_id == self.car_class_id or class_id == self.truck_class_id or self.finetuned:
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].item()
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    bboxes.append((x1, y1, x2, y2))
                    scores.append(confidence)
                    
                    w = x2 - x1
                    h = y2 - y1
                    detections_xywh.append((x1, y1, w, h))
        
        # Store detections with scores (new format)
        detections_with_scores = [(x, y, w, h, conf) for (x, y, w, h), conf in zip(detections_xywh, scores)]
        self.detections[frame_id] = detections_with_scores
        self.scores[frame_id] = scores  # Keep for backward compatibility
        
        return bboxes, scores
