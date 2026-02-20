import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from models import GaussianModel
from detectors import CCDetector
from utils import save_detections_txt

VIDEO_PATH = "data/AICity_data/train/S03/c010/vdo.avi"
RESULTS_PATH = "results/gaussian_k_8_cc_100"
BG_PERCENTAGE = 0.25
MIN_CC_PIXELS = 100
K=8

cap = cv.VideoCapture(VIDEO_PATH)

os.makedirs(RESULTS_PATH, exist_ok=True)

frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

mask_out = cv.VideoWriter(os.path.join(RESULTS_PATH, "mask.avi"), 
                          cv.VideoWriter_fourcc(*'XVID'), 
                          cap.get(cv.CAP_PROP_FPS), 
                          frame_size,
                          isColor=False
                          )

total_frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
bg_frame_num = int(BG_PERCENTAGE * total_frame_num)
processed_frames = 0
bg_frames = []
masks = []

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
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,10))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    mask_out.write(mask)
    masks.append(mask)
    
    frame_id = processed_frames - bg_frame_num
    detector.detect(mask, frame_id)
    processed_frames += 1

cap.release()
mask_out.release()

print("WARNING: THE FRAME IDs START FROM 0 IN THE DETECTIONS, NOT FROM THE ORIGINAL VIDEO")
save_detections_txt(detector.detections, os.path.join(RESULTS_PATH, "detections.txt"))

cap = cv.VideoCapture(VIDEO_PATH)
bbox_out = cv.VideoWriter(os.path.join(RESULTS_PATH, "detections.avi"), 
                          cv.VideoWriter_fourcc(*'XVID'), 
                          cap.get(cv.CAP_PROP_FPS), 
                          frame_size,
                          isColor=True)

frame_count = 0
while True:
    ret, frame = cap.read()    
    if not ret:
        break
    
    if frame_count < bg_frame_num:
        frame_count += 1
        continue
    
    frame_id = frame_count - bg_frame_num
    if frame_id in detector.detections:
        for x, y, w, h in detector.detections[frame_id]:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    bbox_out.write(frame)
    frame_count += 1

cap.release()
bbox_out.release()