import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from models import GaussianModel

VIDEO_PATH = "data/AICity_data/train/S03/c010/vdo.avi"
RESULTS_PATH = "results/gaussian"
BG_PERCENTAGE = 0.25

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

model = GaussianModel(image_size=frame_size)

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

cap.release()
mask_out.release()

