"""
Dataset class for the AI City Challenge dataset
"""

from torch.utils.data import Dataset
import cv2
import utils
import torch
from config import *
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

class AICityDataset(Dataset):
    """
    AI City dataset custom for C6 project files and YOLO usage
    """
    
    def __init__(
        self,
        video_path : str,
        annotation_path : str,
        evaluation : bool = False,
        car_class : int = 2
    ):
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.gt_data, self.annotations = self._extract_frame_annotations(annotation_path, (self.width, self.height))
        self.evaluation_mode = evaluation
        self.car_class = car_class
    
    def __getitem__(self, index : int):
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise IndexError(f"Frame {index} not found")
        
        if self.evaluation_mode:
            
            return frame, self.annotations[index]
        
        coco_bboxes = self.annotations[index]["annotations"]
        
        yolo_boxes = []
        yolo_classes = []
        
        for bbox in coco_bboxes:
            xmin, ymin, xmax, ymax = bbox['bbox']
            x_c = ((xmin + xmax) / 2) / self.width
            y_c = ((ymin + ymax) / 2) / self.height
            w = (xmax - xmin) / self.width
            h = (ymax - ymin) / self.height
            yolo_boxes.append([x_c, y_c, w, h])
            yolo_classes.append(self.car_class)
            
        yolo_boxes = torch.tensor(yolo_boxes, dtype=torch.float32)
        yolo_classes = torch.tensor(yolo_classes, dtype=torch.float32)
            
        return frame, {"bboxes" : yolo_boxes, "classes" : yolo_classes}
        
    
    def __len__(self) -> int:
        
        return self.total_frames
    
    # ------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------
    
    def get_gt_dict(self):
        return self.annotations
    
    def create_evaluator(self) -> COCOEvaluator:
        
        def get_dataset():
            return self.gt_data
        
        if "video_dataset" not in DatasetCatalog:
            DatasetCatalog.register("video_dataset", get_dataset)
        
        MetadataCatalog.get("video_dataset").set(thing_classes=["car"])

        evaluator = COCOEvaluator("video_dataset", output_dir="./results/COCO_output")
        evaluator.reset()
        
        return evaluator
    
    # ------------------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------------------
    
    @staticmethod
    def _extract_frame_annotations(annotation_path : str, image_size : tuple):
        
        gt_data = utils.get_COCO_gt(annotation_path, image_size)
        
        gt_dict = {d["image_id"]: d for d in gt_data}
        
        return gt_data, gt_dict
    
    

if __name__ == "__main__":
    
    args = utils.set_args()
    config = build_config(args, "yolo_run")
    dataset = AICityDataset(video_path=config.input_path, annotation_path=config.xml_path, evaluation=True)
    
    print(dataset[212][1])