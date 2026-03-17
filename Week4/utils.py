import argparse

def set_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--data", help="Data folder containing Video", default="../AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c005", type=str)
    parse.add_argument("-r", "--results", help="Folder to leave the results", default="results/S01/c005", type=str)
    
    return parse.parse_args()

def save_detections_txt(detections, filepath):
    """
    Save detections to a text file with confidence scores.
    
    Args:
        detections: Dictionary mapping frame_id to list of (x, y, w, h, confidence) tuples
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for frame_id in sorted(detections.keys()):
            for x, y, w, h, conf in detections[frame_id]:
                f.write(f"{frame_id},{x},{y},{w},{h},{conf}\n")
                
                
        
                
import xml.etree.ElementTree as ET
from typing import Tuple
import argparse


def read_annotations_xml(xml_path : str) -> dict:
    """
    Read car annotations from XML file.
    
    Unlike Week1, this function:
    - Only extracts cars (label="car")
    - Accepts ALL cars regardless of parked status
    
    Args:
        xml_path: Path to the XML annotation file
        
    Returns:
        Dictionary mapping frame_id to list of bounding boxes (x, y, w, h)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    
    for track in root.findall('track'):
        # Check if this track is a car
        label = track.get('label')
        if label != 'car':
            continue
            
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            ytl = int(float(box.get('ytl')))
            xtl = int(float(box.get('xtl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
         
            if frame not in annotations:
                annotations[frame] = [] 
         
            annotations[frame].append((xtl, ytl, xbr, ybr))
    
    return annotations





def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Tuple (x1, y1, x2, y2) in XYXY format
        box2: Tuple (x1, y1, x2, y2) in XYXY format
        
    Returns:
        IoU value (float between 0 and 1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def save_detections_txt(detections, filepath):
    """
    Save detections to a text file with confidence scores.
    
    Args:
        detections: Dictionary mapping frame_id to list of (x, y, w, h, confidence) tuples
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for frame_id in sorted(detections.keys()):
            for x, y, w, h, conf in detections[frame_id]:
                f.write(f"{frame_id},{x},{y},{w},{h},{conf}\n")


def load_detections_txt(filepath):
    """
    Load detections from a text file.
    
    Args:
        filepath: Input file path with format: frame_id,x,y,w,h,confidence
        
    Returns:
        Dictionary mapping frame_id to list of (x, y, w, h, confidence) tuples
    """
    detections = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0  # Default confidence if not present
            
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append((x, y, w, h, conf))
    
    return detections


def save_tracking_txt(tracks, filepath):
    """
    Save tracking results to a text file in MOTChallenge format.
    
    Args:
        tracks: Dictionary mapping frame_id to list of (track_id, bbox) tuples
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for frame_id in sorted(tracks.keys()):
            for track_id, bbox in tracks[frame_id]:
                x, y, w, h = bbox[:4]  # Ignore confidence score if present
                # MOTChallenge format: frame, id, x, y, w, h, conf, -1, -1, -1
                f.write(f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n")


def set_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--data", help="Data folder containing AICity_data", default="../data/", type=str)
    parse.add_argument("-r", "--results", help="Folder to leave the results", default="results", type=str)
    
    return parse.parse_args()
