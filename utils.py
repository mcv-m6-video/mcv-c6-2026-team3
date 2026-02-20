import xml.etree.ElementTree as ET


#Read the annotations from Team 1 2018/2019
def read_annotations_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    
    for track in root.findall('track'):
        for box in track.findall('box'):
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


def save_detections_txt(detections, filepath):
    with open(filepath, 'w') as f:
        for frame_id in sorted(detections.keys()):
            for x, y, w, h in detections[frame_id]:
                f.write(f"{frame_id},{x},{y},{w},{h}\n")
