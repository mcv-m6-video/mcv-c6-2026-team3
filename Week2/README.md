# C6 Team3 - Week 2

## Project structure
```
Week2
├── config         <--- Configuration class for handling paths
├── datasets       <--- Dataset classes for AI City dataset
├── detectors      <--- YOLO detector for object detection
├── models         <--- Trained YOLO models
├── pipelines      <--- Complete pipelines for detection and tracking
├── plots          <--- Plotter notebooks for visualization
├── results        <--- Output results from detection and tracking
├── run            <--- Our best templates for running detection and tracking
├── studies        <--- Studies programs for hyperparameter search
├── trackers       <--- Tracking algorithms (ByteTrack, IOU, OCSORT, SORT)
└── utils          <--- Utility functions
```

## Installation

Execute the following commands:

```bash
conda create -n c6 python=3.11 -y
conda activate c6
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

## Task 1: Object Detection with YOLO

In order to run YOLO detection you need to do:

```bash
python run/run_yolo.py -d data_folder -r result_folder
```

The data folder must contain the AICity dataset and the annotations file within it.

To change the YOLO model, open the run script and modify the `YOLO_MODEL` parameter. Available models are located in the `models/` directory.

The results will include:
- A video with bounding boxes for detected objects
- A `detections.txt` file with all detections
- A `metrics.txt` file containing mAP@0.5, mAP@0.75, and other metrics from detectron2

### Fine-tuning YOLO

To fine-tune YOLO on the AI City dataset:

```bash
python run/finetune_25.py -d data_folder -r result_folder
```

This will perform k-fold cross-validation and save the best model.

## Task 2: Multi-Object Tracking

After running detection (Task 1), you can perform tracking using different trackers:


### IOU Tracker

```bash
python run/run_tracking_iou.py -d data_folder -r result_folder
```

### SORT

```bash
python run/run_tracking_sort.py -d data_folder -r result_folder
```

### OCSORT

```bash
python run/run_tracking_ocsort.py -d data_folder -r result_folder
```

### ByteTrack

```bash
python run/run_tracking_bytetrack.py -d data_folder -r result_folder
```



**Note:** All tracking scripts require the `detections.txt` file generated from Task 1. Make sure to run detection first.

The tracking results will include:
- A video with bounding boxes and tracking IDs
- TrackEval metrics HOTA and IDF1
- A tracking output file compatible with MOTChallenge format

To change tracker parameters, modify the configuration variables at the beginning of each run script in the `run/` directory. The parameters in this directory are the best found via hyperparameter optimization.
