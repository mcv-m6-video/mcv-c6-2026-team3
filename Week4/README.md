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

### Online MTMC (main.py)

Per-frame tracking with SORT or ByteTrack, augmented with a cross-camera ReID matcher that gates proposals using GPS distance, travel speed, and appearance (histogram or Siamese embeddings). Cameras are time-synchronised via `cam_timestamp` offsets.

```bash
# Run on Sequence 3 with finetuned detections and evaluate with AIC metric
python main.py \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --dets_dir results/finetuned/S03 \
    --tracker bytetrack \
    --scorer siamese --siamese_ckpt checkpoints/siamese_best.pth \
    --conf_thr 0.3 --use_roi \
    --eval --aic_eval
```

Grid search over matching hyperparameters:

```bash
python main.py \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --dets_dir results/finetuned/S03 \
    --grid_search --metrics_csv results/online_gs.csv \
    --gs_lookback 10 20 --gs_geo_sigma 30 100 --gs_conf_thr 0.3 0.4
```

### Offline MTMC (offline_mtmc.py)

Runs local tracking per camera, builds **tracklets** with GPS velocity profiles, filters parked vehicles by pixel displacement, then matches tracklets across cameras with a greedy algorithm scoring temporal overlap, GPS proximity, velocity similarity, and appearance.

```bash
# Best configuration found via grid search on S03
python offline_mtmc.py \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --dets_dir results/finetuned/S03 \
    --conf_thr 0.3 \
    --match_thr 0.4 \
    --max_time_gap 60.0 \
    --max_speed 30.0 \
    --geo_sigma 100.0 \
    --w_geo 0.5 --w_vel 0.2 --w_app 0.4 \
    --eval --aic_eval \
    --montage --write_video
```

Grid search over all matching parameters:

```bash
python offline_mtmc.py \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --dets_dir results/finetuned/S03 \
    --grid_search --gs_csv results/offline/S03/gs_results.csv \
    --gs_match_thr 0.3 0.4 0.5 \
    --gs_geo_sigma 30 50 100 \
    --gs_w_geo 0.3 0.4 0.5 \
    --gs_w_app 0.3 0.4 0.5
```

### Per-car GPS visualisation (track_visualizer.py)

Generates a video for a chosen global car ID showing a live GPS dot on an OpenStreetMap tile (left) and a synchronised camera grid with bounding boxes (right).

```bash
# List all available global IDs and how many cameras each appears in
python track_visualizer.py \
    --out_dir results/offline/S03 \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --list_ids

# Render car ID 2
python track_visualizer.py \
    --out_dir results/offline/S03 \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --car_id 2 --output results/offline/S03/car_2.mp4
```

### Detector comparison (compare_detectors.py)

Runs the full MTMC pipeline for each detector variant (default / finetuned / large) across all sequences and prints a comparison table.

```bash
python compare_detectors.py \
    --scenario_dirs ../AI_CITY_CHALLENGE_2022_TRAIN/train/S01 \
                    ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --conf_thr 0.3 --use_roi
```

### GPS ROI map (plot_roi_gps.py)

Projects each camera's region-of-interest mask onto a shared GPS map to verify geographic coverage.

```bash
python plot_roi_gps.py \
    --scenario_dir ../AI_CITY_CHALLENGE_2022_TRAIN/train/S03 \
    --output map_roi_s03.png
```
