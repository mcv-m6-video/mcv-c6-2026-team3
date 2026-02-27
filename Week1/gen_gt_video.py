import cv2 as cv
import os
from utils import read_annotations_xml

VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"
XML_PATH = "../data/ai_challenge_s03_c010-full_annotation.xml"
OUTPUT_PATH = "results/gt_video.avi"

gt_annotations = read_annotations_xml(XML_PATH)

cap = cv.VideoCapture(VIDEO_PATH)
frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv.CAP_PROP_FPS)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

video_out = cv.VideoWriter(OUTPUT_PATH, 
                          cv.VideoWriter_fourcc(*'XVID'), 
                          fps, 
                          frame_size,
                          isColor=True)

frame_id = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_id in gt_annotations:
        for x, y, w, h in gt_annotations[frame_id]:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    video_out.write(frame)
    frame_id += 1

cap.release()
video_out.release()
