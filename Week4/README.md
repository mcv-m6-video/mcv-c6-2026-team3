# C6 Team3 - Week 4

## Project structure
```
Week4
├── checkpoints             <--- Siamese model checkpoints
├── config                  <--- Configuration class for handling paths
├── detectors               <--- YOLO detector for object detection
├── evaluation              <--- Tracking evaluation utilities
├── models                  <--- Trained YOLO models
├── outputs                 <--- Output results (generated at runtime)
├── pipelines               <--- Detection pipeline
├── trackers                <--- ByteTrack tracker implementation
└── utils                   <--- Utility functions
```

## Installation

### Base environment

Execute the following commands:

```bash
conda create -n c6w4 python=3.11 -y
conda activate c6w4
pip install -r requirements.txt
```

## Siamese-based ReID Approach

### Extract Ground Truth Crops

Extract vehicle crops from ground truth annotations for training the Siamese network.

```bash
python extract_gt_crops.py \
    --data_root /path/to/aic22-track1 \
    --output_dir gt_crops
```

### Train Siamese ReID Network

Train a ResNet-18 based Siamese network using triplet loss for vehicle re-identification.

```bash
python train_resnet_siamese.py \
    --data_root gt_crops \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --margin 0.3 \
    --output_dir checkpoints
```

### Generate YOLO Detections

Generate detection files from YOLO for all sequences and cameras.

```bash
python generate_detections.py \
    --data_root /path/to/aic22-track1 \
    --output_dir outputs \
    --model_path models/yolo11x_finetune.pt \
    --conf_thr 0.25
```

### Multi-Camera Tracking with ReID

Run ByteTrack with Siamese ReID embeddings for multi-camera tracking.

```bash
python run_multicamera_bytetrack_reid.py \
    --data_root /path/to/aic22-track1 \
    --detections_root outputs \
    --siamese_ckpt checkpoints/siamese_best.pth \
    --output_dir outputs/tracking_bytetrack_reid \
    --use_reid \
    --use_mtmc \
    --track_thresh 0.25 \
    --match_thresh 0.8 \
    --intra_cam_thr 0.85 \
    --inter_cam_thr 0.65
```

## GPS-based Tracking Approaches
