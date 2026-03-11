# C6 Team3 - Week 3

## Project structure
```
Week3
├── config                  <--- Configuration class for handling paths
├── detectors               <--- YOLO detector for object detection
├── evaluation              <--- Tracking evaluation utilities (HOTA, IDF1)
├── gmflow                  <--- GMFlow optical flow model (not in repo, see installation)
├── NeuFlow_v2              <--- NeuFlow v2 optical flow model (not in repo, see installation)
├── NeuFlow_v2_weights      <--- NeuFlow v2 weights (not in repo, see installation)
├── FlowFormerPlusPlus      <--- FlowFormer++ optical flow model (not in repo, see installation)
├── models                  <--- Trained YOLO models
├── pipelines               <--- Detection pipeline
├── results                 <--- Output results from tracking (generated at runtime, not in repo)
└── utils.py                <--- Utility functions
```

## Installation

### Base environment

Execute the following commands:

```bash
conda create -n c6w3 python=3.11 -y
conda activate c6w3
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

### External optical flow dependencies

> **Note:** All model directories and weights are **not included in this repository**. You need to clone each repository and download the corresponding checkpoints manually before running any script.

- **PyFlow**: [https://github.com/pathak22/pyflow](https://github.com/pathak22/pyflow) — clone into `pyflow/`, then compile the Cython extension with `python setup.py build_ext -i` inside it
- **GMFlow**: [https://github.com/haofeixu/gmflow](https://github.com/haofeixu/gmflow) — clone into `gmflow/`, install dependencies from `gmflow/environment.yml`, and download the desired checkpoint (e.g. `gmflow_kitti-285701a8.pth`) into `gmflow/checkpoints/`
- **FlowFormer++**: [https://github.com/XiaoyuShi97/FlowFormerPlusPlus](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) — clone into `FlowFormerPlusPlus/`, follow its README, and download the KITTI checkpoint to `FlowFormerPlusPlus/checkpoints/kitti.pth`
- **NeuFlow v2**: [https://github.com/neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2) — clone into `NeuFlow_v2/`, and download the weights to `NeuFlow_v2_weights/model.safetensors`

## Task 1: Optical Flow Estimation

Evaluate and compare multiple optical flow methods (PyFlow, FlowFormer++, NeuFlow v2, GMFlow) on KITTI ground truth sequences, computing MSEN and PEPN metrics.

```bash
python run_optical_flow.py \
    --ff_ckpt FlowFormerPlusPlus/checkpoints/kitti.pth \
    --gmflow_ckpt gmflow/checkpoints/gmflow_kitti-285701a8.pth
```


## Task 2: Optical-Flow-Guided Multi-Object Tracking

Tracking is performed on top of YOLO detections (`detections.txt`) using optical flow to improve bounding box propagation between frames.

### Generate detections (if not already available)

```bash
python run_yolo.py --data data_folder --results result_folder
```

### Flow-IOU Tracker

IOU-based tracker where lost tracks are propagated using FlowFormer++ optical flow.

```bash
python run_flow_tracking.py \
    --video path/to/vdo.avi \
    --dets detections.txt \
    --ff_ckpt FlowFormerPlusPlus/checkpoints/kitti.pth \
    --out results/tracking_flow.mp4 \
    --iou_thr 0.05 \
    --max_age 5
```

### SORT + Flow Tracker

Hybrid tracker combining SORT (Kalman filter) with FlowFormer++ optical flow predictions, blended via an `--alpha` parameter (`0.0` = 100% flow, `1.0` = 100% Kalman).

```bash
python run_sort_flow_tracking.py \
    --video path/to/vdo.avi \
    --dets detections.txt \
    --ff_ckpt FlowFormerPlusPlus/checkpoints/kitti.pth \
    --out results/tracking_sort_flow.mp4 \
    --iou_thr 0.1458 \
    --max_age 5 \
    --min_hits 12 \
    --alpha 0.5
```

### ByteTrack + Flow Tracker

ByteTrack extended with FlowFormer++ optical flow propagation.

```bash
python run_bytetrack_flow.py \
    --video path/to/vdo.avi \
    --dets detections.txt \
    --ff_ckpt FlowFormerPlusPlus/checkpoints/kitti.pth \
    --out results/tracking_bytetrack_flow.mp4 \
    --track_thresh 0.6859 \
    --match_thresh 0.6519 \
    --max_age 60 \
    --alpha 0.1
```

### Run all sequences (Task 2 pipeline)

To run detection and tracking across all sequences (S01, S03, S04) automatically:

```bash
python task_2.py
```

This file checks what results have been computed from the dataset. For those that are missing, it first runs yolo to create the detection files, and then runs both run_flow_tracking_script.py and run_sort_flow_tracking_script.py for that sequence.
