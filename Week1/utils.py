import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from typing import Tuple
import argparse
import optuna


#Read the annotations from Team 1 2018/2019
def read_annotations_xml(xml_path : str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    
    for track in root.findall('track'):
        for box in track.findall('box'):
            
            # Default value
            parked = False
            
            # Find attribute named "parked"
            for attr in box.findall("attribute"):
                if attr.get("name") == "parked":
                    parked = attr.text.lower() == "true"  # True/False
            
            if parked:
                continue
            
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            x = int(xtl)
            y = int(ytl)
            w = int(xbr - xtl)
            h = int(ybr - ytl)
            
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append((x, y, w, h))
    
    return annotations

def get_COCO_gt(xml_path : str, image_size : Tuple[int, int], init_frame : int) -> dict:
    
    annotations = read_annotations_xml(xml_path)
    
    width, height = image_size
    
    gt_data = []
    
    max_frames = max(annotations.keys())
    
    for frame_id in range(max_frames + 1):
        if frame_id < init_frame:
            continue
        
        if not annotations.get(frame_id, []):
            gt_data.append({
                "file_name" : f"frame_{frame_id}.jpg",
                "image_id" : frame_id,
                "height" : height,
                "width" : width,
                "annotations" : []
            })
            continue
        
        gt_boxes = annotations[frame_id]
        frame_annotations = []
        for box in gt_boxes:
            x, y, w, h = box
            frame_annotations.append({
                "bbox" : [x, y, w, h],
                "bbox_mode" : BoxMode.XYWH_ABS,
                "category_id" : 0
            })
        
        gt_data.append({
            "file_name" : f"frame_{frame_id}.jpg",
            "image_id" : frame_id,
            "height" : height,
            "width" : width,
            "annotations" : frame_annotations
        })
        
    return gt_data

def save_detections_txt(detections, filepath):
    with open(filepath, 'w') as f:
        for frame_id in sorted(detections.keys()):
            for x, y, w, h in detections[frame_id]:
                f.write(f"{frame_id},{x},{y},{w},{h}\n")

def set_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--data", help="Data folder containing AICity_data", default="../data/", type=str)
    parse.add_argument("-r", "--results", help="Folder to leave the results", default="results", type=str)
    parse.add_argument("-k", "--k", help="Deviation multiplier",default=2.5, type=float)
    parse.add_argument("-m", "--min", help="Minimum ammount of pixels for connected component", default=100, type=int)
    
    return parse.parse_args()


class ConvergenceEarlyStopping:
    def __init__(self, patience: int = 100, tolerance: float = 1e-3):
        self.patience = patience
        self.tolerance = tolerance
        self.best_score = -float('inf')
        self.wait = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        current_value = study.best_value
        
        if current_value > self.best_score + self.tolerance:
            self.best_score = current_value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            print(f"\n[Early Stopping] Convergence reached. No improvement greater than {self.tolerance} for {self.patience} consecutive trials.")
            study.stop()