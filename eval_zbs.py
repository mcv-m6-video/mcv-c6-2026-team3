import cv2 as cv
import numpy as np  
import os
from pathlib import Path
import argparse
from detectors import CCDetector
from utils import *
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
import torch
import contextlib
from preprocess import * 

def create_evaluator(annotations : Path, frame_size : Tuple[int, int], initial_frame : int) -> Tuple[COCOEvaluator, dict]:
        
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

parser = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
parse.add_argument("-i", "--input", help="Input mask video resulting from ZBS", required=True, type=str)
parse.add_argument("-r", "--results", help="Folder to leave the results", default="results", type=str)
parse.add_argument("-a", "--annotations", help="Annotation file with the bounding boxes", default="data/ai_challenge_s03_c010-full_annotation.xml", type=str)
parse.add_argument("-d", "--data", help="Data folder containing AICity_data", default="data/", type=str)
args = parse.parse_args()

data_path = Path(f"{args.data}/AICity_data/train/S03/c010/vdo.avi")

input_path = args.input
annotatations_path = args.annotations
result_path = args.results

os.makedirs(result_path, exist_ok=True)

cap = cv.VideoCapture(input_path)
aux_cap = cv.VideoCapture(data_path)
detector = CCDetector(min_pixels=500)

height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

frame_size = (width, height)

preprocess_fn = preprocess.generate_morph_func_adap(4, (20, 45))

bbox_out = cv.VideoWriter(os.path.join(result_path, "detections.avi"), 
                                    cv.VideoWriter_fourcc(*'XVID'), 
                                    cap.get(cv.CAP_PROP_FPS), 
                                    frame_size,
                                    isColor=True)
    
total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
bg_frame_num = 0
processed_frames = 0
bg_frames = []

evaluator, gt_dict = create_evaluator(annotatations_path, frame_size, bg_frame_num)
        
for i in range(216):
    _, _ = aux_cap.read()
        
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, binary_frame = cv.threshold(grayscale_frame, 127, 255, cv.THRESH_BINARY)
    
    ret, frame = aux_cap.read()
    
    bboxes, detection = detector.detect(binary_frame, processed_frames, None)
    
    instances = Instances((height, width))
    instances.pred_boxes = Boxes(bboxes)
    instances.scores = torch.ones(len(bboxes)) * 0.99
    instances.pred_classes = torch.zeros(len(bboxes))
    
    prediction = {
        "image_id" : processed_frames + 216,
        "instances" : instances
    }
    
    evaluator.process([gt_dict[processed_frames + 216]], [prediction])
    
    for x1, y1, x2, y2 in bboxes:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    bbox_out.write(frame)
    
    processed_frames += 1
    
print(processed_frames)
print(len(gt_dict))

cap.release()
bbox_out.release()
bbox_out.release()


with open(f"{result_path}/metrics.txt", "w") as f, contextlib.redirect_stdout(f):
    results = evaluator.evaluate()

with open(f"{result_path}/metrics.txt", "a") as f:
    f.write("\n")
    f.write("--------------------------------------------------------------------------------\n")
    f.write("\n")
    f.write(f"mAP@05 : {results['bbox']['AP50']}")

print(f"Results can be found inside {result_path} folder")