import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
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


def get_COCO_gt(xml_path : str, image_size : Tuple[int, int], init_frame : int = 0) -> dict:
    """
    Generate COCO-format ground truth data from XML annotations.
    
    Args:
        xml_path: Path to the XML annotation file
        image_size: Tuple of (width, height)
        init_frame: Initial frame number to start from (default: 0)
        
    Returns:
        List of dictionaries in COCO format
    """
    annotations = read_annotations_xml(xml_path)
    
    width, height = image_size
    
    gt_data = []
    
    max_frames = max(annotations.keys()) if annotations else 0
    
    for frame_id in range(init_frame, max_frames + 1):
        
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
            x1, y1, x2, y2 = box
            frame_annotations.append({
                "bbox" : [x1, y1, x2, y2],
                "bbox_mode" : BoxMode.XYXY_ABS,
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
    """
    Save detections to a text file.
    
    Args:
        detections: Dictionary mapping frame_id to list of bounding boxes
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for frame_id in sorted(detections.keys()):
            for x, y, w, h in detections[frame_id]:
                f.write(f"{frame_id},{x},{y},{w},{h}\n")


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
