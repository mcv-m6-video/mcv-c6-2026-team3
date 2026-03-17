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
        