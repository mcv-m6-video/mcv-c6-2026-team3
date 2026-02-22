import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from models import GaussianModel
from detectors import CCDetector
from utils import *
import time
import torch
from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog


arg = set_args()

XML_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
K = arg.k
MIN_CC_PIXELS = arg.min
VIDEO_PATH = f"{arg.data}/AICity_data/train/S03/c010/vdo.avi"
RESULTS_PATH = f"{arg.results}/gaussian_k_{K}_cc_{MIN_CC_PIXELS}"
BG_PERCENTAGE = 0.25

os.makedirs(RESULTS_PATH, exist_ok=True)

cap = cv.VideoCapture(VIDEO_PATH)

height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

frame_size = (width, height)


mask_out = cv.VideoWriter(os.path.join(RESULTS_PATH, "mask.avi"), 
                          cv.VideoWriter_fourcc(*'XVID'), 
                          cap.get(cv.CAP_PROP_FPS), 
                          frame_size,
                          isColor=False
                          )
bbox_out = cv.VideoWriter(os.path.join(RESULTS_PATH, "detections.avi"), 
                          cv.VideoWriter_fourcc(*'XVID'), 
                          cap.get(cv.CAP_PROP_FPS), 
                          frame_size,
                          isColor=True)

total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
bg_frame_num = int(BG_PERCENTAGE * total_frame_num)
processed_frames = 0
bg_frames = []


gt_data = get_COCO_gt(XML_PATH, frame_size, bg_frame_num)
def get_dataset():
    return gt_data

DatasetCatalog.register("video_dataset", get_dataset)
MetadataCatalog.get("video_dataset").set(thing_classes=["object"])

evaluator = COCOEvaluator("video_dataset", output_dir="./results/COCO_output")
evaluator.reset()

gt_dict = {d["image_id"]: d for d in gt_data}

model = GaussianModel(image_size=frame_size, K=K)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

times = {}
predictions = []

while True:

    ret, frame = cap.read()
    
    if not ret:
        break


    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if processed_frames < bg_frame_num:
        processed_frames += 1
        bg_frames.append(grey_frame)
        if processed_frames == bg_frame_num:
            model.modelize_back(np.array(bg_frames))
            
        continue
    
    mask = model(grey_frame).astype(np.uint8) * 255
 
    bboxes, detection = detector.detect(mask, processed_frames, preprocess=True) 

    instances = Instances((height, width))
    instances.pred_boxes = Boxes(bboxes)
    instances.scores = torch.ones(len(bboxes)) * 0.99
    instances.pred_classes = torch.zeros(len(bboxes))
    
    predictions.append({
        "image_id" : processed_frames,
        "instances" : instances
    })
    
    evaluator.process([gt_dict[processed_frames]], [predictions[-1]])   

    mask_out.write(detection)
    
    for x1, y1, x2, y2 in bboxes:
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    bbox_out.write(frame)
    
    processed_frames += 1

cap.release()
mask_out.release()
bbox_out.release()

results = evaluator.evaluate()

with open(f"{RESULTS_PATH}/metrics.txt", "w") as f:
    f.write(f"mAP@05 : {results['bbox']['AP50']}")
    print(f"mAP@05 : {results['bbox']['AP50']}")


save_detections_txt(detector.detections, os.path.join(RESULTS_PATH, "detections.txt"))