# C6 Team3 - Week 2

## Project Overview

### Task 1

Task 1 focuses on **object detection evaluation** using YOLO models on traffic videos. Unlike Week 1's background/foreground detection, this week evaluates YOLO's vehicle detection performance using metrics like IoU, mAP@0.5, and mAP@0.75.

**Detected Classes**: Cars and trucks (COCO classes 2 and 7)  
**Ground Truth**: Only cars (label="car"), including both parked and non-parked vehicles

A percentage of train frames are set, those are not used for evaluation.



## Project Structure

```
Week2/
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration management
├── detectors/
│   ├── __init__.py
│   └── yolo_detector.py       # YOLO-based car detector
├── pipelines/
│   ├── __init__.py
│   └── detection_pipeline.py  # Detection evaluation pipeline
├── run/
│   ├── __init__.py
│   └── run_yolo.py            # Main script to run YOLO detection
├── models/                    # YOLO models (.pt files, auto-downloaded)
├── results/
│   └── yolo_run/              # Output folder (created on run)
│       ├── detections.avi     # Video with detections
│       ├── detections.txt     # Frame-by-frame detections
│       └── metrics.txt        # Evaluation metrics
├── utils.py                   # Utility functions (GT parsing, etc.)
└── README.md
```


## Installation

Execute the following commands:

```bash
conda create -n c6 python=3.11 -y
conda activate c6
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install ultralytics
```

## Usage

### Run YOLO Detection

From the `Week2` directory:

```bash
python run/run_yolo.py -d ../data/ -r results
```

### Command-line Arguments

- `-d, --data`: Path to data folder containing `AICity_data` (default: `../data/`)
- `-r, --results`: Path to results folder (default: `results`)

### Output

The script generates:

1. **detections.avi**: Video with bounding boxes
   - Red boxes: Ground truth
   - Green boxes: YOLO predictions (with confidence scores)

2. **detections.txt**: Text file with detections in format:
   ```
   frame_id,x,y,w,h
   ```

3. **metrics.txt**: Evaluation metrics including:
   - mAP@0.5 (IoU threshold = 0.5)
   - mAP@0.75 (IoU threshold = 0.75)
   - mAP (averaged over IoU thresholds)
   - mIoU (mean Intersection over Union)

## Code Structure

### Modular Design

The code follows a highly modular architecture similar to Week 1:

- **config/**: Manages paths and configuration
- **detectors/**: YOLO detector class (filters cars and trucks - COCO classes 2 and 7)
- **pipelines/**: End-to-end detection and evaluation pipeline with mIoU calculation
- **utils.py**: Ground truth parsing, COCO dataset creation, file I/O

### Example: Running with Different YOLO Models

Edit `run/run_yolo.py` and change the `YOLO_MODEL` variable:

```python
YOLO_MODEL = "yolo26n.pt"  # Nano (fastest)
YOLO_MODEL = "yolo26s.pt"  # Small
YOLO_MODEL = "yolo26m.pt"  # Medium
YOLO_MODEL = "yolo26l.pt"  # Large
```

## Evaluation Metrics

- **mIoU (mean IoU)**: Average IoU across all frames for matched predictions (IoU ≥ 0.5)
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- **mAP**: Mean Average Precision averaged over IoU thresholds [0.5:0.05:0.95]

### Important Notes

- YOLO detects both **cars** and **trucks** (COCO classes 2 and 7)
- Ground truth only includes **cars** (label="car" in XML)
- Confidence scores from YOLO are used for ranking in mAP calculation
- mIoU is calculated manually by matching predictions to ground truth boxes

