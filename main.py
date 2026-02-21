import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from models import GaussianModel
from detectors import CCDetector
from utils import save_detections_txt
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-d", "--data", help="Data folder containing AICity_data", default="data/", type=str)
parse.add_argument("-r", "--results", help="Folder to leave the results", default="results", type=str)
parse.add_argument("-k", "--k", help="Deviation multiplier",default=2.5, type=float)
parse.add_argument("-m", "--min", help="Minimum ammount of pixels for connected component", default=100, type=int)
arg = parse.parse_args()


K = arg.k
MIN_CC_PIXELS = arg.min
VIDEO_PATH = f"{arg.data}/AICity_data/train/S03/c010/vdo.avi"
RESULTS_PATH = f"{arg.results}/gaussian_k_{K}_cc_{MIN_CC_PIXELS}"
BG_PERCENTAGE = 0.25

os.makedirs(RESULTS_PATH, exist_ok=True)

cap = cv.VideoCapture(VIDEO_PATH)

frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

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

model = GaussianModel(image_size=frame_size, K=K)
detector = CCDetector(min_pixels=MIN_CC_PIXELS)

while True:

    ret, frame = cap.read()
    
    if not ret:
        break
    
    grey_frame = np.array(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    
    if processed_frames < bg_frame_num:
        processed_frames += 1
        bg_frames.append(grey_frame)
        if processed_frames == bg_frame_num:
            model.modelize_back(np.array(bg_frames))
            
        continue
    
    mask = model(grey_frame).astype(np.uint8) * 255
    
    frame_id = processed_frames - bg_frame_num
    
    bboxes, detection = detector.detect(mask, frame_id, preprocess=True) 
    
    mask_out.write(detection)
    
    for x, y, w, h in bboxes:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    bbox_out.write(frame)
    
    processed_frames += 1

cap.release()
mask_out.release()
bbox_out.release()

print("WARNING: THE FRAME IDs START FROM 0 IN THE DETECTIONS, NOT FROM THE ORIGINAL VIDEO")
save_detections_txt(detector.detections, os.path.join(RESULTS_PATH, "detections.txt"))