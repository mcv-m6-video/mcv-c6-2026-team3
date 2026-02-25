import cv2 as cv
import numpy as np
import torch
import os
import contextlib
from pathlib import Path
from typing import Callable, Tuple
from tqdm import tqdm

from detectron2.structures import Boxes, Instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

from utils import get_COCO_gt, save_detections_txt


def _parse_gt_masks(annotations: Path, frame_size: Tuple[int, int],
                    first_frame: int, last_frame: int) -> dict:
    width, height = frame_size
    gt_data = get_COCO_gt(annotations, frame_size, first_frame)

    masks = {}
    for entry in gt_data:
        fid = entry["image_id"]
        if fid < first_frame or fid >= last_frame:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in entry.get("annotations", []):
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        masks[fid] = mask

    return masks


class DLDetectionPipeline:
    def __init__(self, model, detector,
                 preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        self.model = model
        self.detector = detector
        self.preprocess_fn = preprocess_fn

    # ------------------------------------------------------------------
    def _create_evaluator(self, annotations: Path,
                           frame_size: Tuple[int, int],
                           initial_frame: int):
        gt_data = get_COCO_gt(annotations, frame_size, initial_frame)

        dataset_name = "video_dataset_dl"
        if dataset_name not in DatasetCatalog:
            DatasetCatalog.register(dataset_name, lambda: gt_data)
        MetadataCatalog.get(dataset_name).set(thing_classes=["object"])

        evaluator = COCOEvaluator(dataset_name, output_dir="./results/COCO_output_dl")
        evaluator.reset()

        gt_dict = {d["image_id"]: d for d in gt_data}
        return evaluator, gt_dict

    # ------------------------------------------------------------------
    def __call__(self, input: Path, output: Path, annotations: Path,
                 bg_percentage: float, save: bool = True) -> float:

        cap = cv.VideoCapture(str(input))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fps    = cap.get(cv.CAP_PROP_FPS)
        frame_size = (width, height)

        total_frames  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        bg_frame_num  = int(bg_percentage * total_frames)
        eval_frames   = total_frames - bg_frame_num

        gt_masks_dict = _parse_gt_masks(annotations, frame_size, 0, bg_frame_num)

        if save:
            os.makedirs(str(output), exist_ok=True)
            mask_out = cv.VideoWriter(
                os.path.join(str(output), "mask.avi"),
                cv.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=False)
            bbox_out = cv.VideoWriter(
                os.path.join(str(output), "detections.avi"),
                cv.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=True)

        evaluator, gt_dict = self._create_evaluator(annotations, frame_size, bg_frame_num)

        train_frames: list[np.ndarray] = []
        train_masks:  list[np.ndarray] = []

        frame_idx = 0
        with tqdm(total=bg_frame_num, desc="  Reading train frames", unit="fr") as pbar:
            while frame_idx < bg_frame_num:
                ret, frame = cap.read()
                if not ret:
                    break
                train_frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
                gt_mask = gt_masks_dict.get(frame_idx,
                                            np.zeros((height, width), dtype=np.uint8))
                train_masks.append(gt_mask)
                frame_idx += 1
                pbar.update(1)

        self.model.modelize_back(np.array(train_frames), np.array(train_masks))
        del train_frames, train_masks, gt_masks_dict

        processed_frames = bg_frame_num

        with tqdm(total=eval_frames, desc="  Inference", unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                mask = self.model(grey).astype(np.uint8) * 255

                bboxes, detection = self.detector.detect(mask, processed_frames, self.preprocess_fn)

                instances = Instances((height, width))
                instances.pred_boxes   = Boxes(bboxes)
                instances.scores       = torch.ones(len(bboxes)) * 0.99
                instances.pred_classes = torch.zeros(len(bboxes), dtype=torch.int64)

                prediction = {"image_id": processed_frames, "instances": instances}

                if processed_frames in gt_dict:
                    evaluator.process([gt_dict[processed_frames]], [prediction])

                if save:
                    mask_out.write(detection)
                    for x1, y1, x2, y2 in bboxes:
                        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    bbox_out.write(frame)

                processed_frames += 1
                pbar.update(1)

        cap.release()

        if save:
            mask_out.release()
            bbox_out.release()
            with open(f"{output}/metrics.txt", "w") as f, contextlib.redirect_stdout(f):
                results = evaluator.evaluate()
            with open(f"{output}/metrics.txt", "a") as f:
                f.write(f"\nmAP@0.5 : {results['bbox']['AP50']}\n")
            save_detections_txt(self.detector.detections, os.path.join(str(output), "detections.txt"))
            print(f"[DLPipeline] Results saved to {output}")
            return results['bbox']['AP50']

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            results = evaluator.evaluate()
        return results['bbox']['AP50']
