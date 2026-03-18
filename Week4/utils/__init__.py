"""
Week4 utility package.
"""

def compute_iou(box1, box2):
    """IoU between two XYXY boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2); y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2); y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
    return inter / union if union > 0 else 0.0
