#!/usr/bin/env python3
"""
File containing the main training script for T-DEED / Task 1 baseline.

Modified version:
- adds optional Weighted BCE
- computes class-wise pos_weight from stored train labels.pkl
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from tabulate import tabulate
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model.model_classification_mod import Model
from model.model_classification_temporal import ModelTemporal
from thop import profile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    args.frame_dir = config["frame_dir"]
    args.save_dir = config["save_dir"] + "/" + args.model
    args.store_dir = config["save_dir"] + "/" + "splits"
    args.labels_dir = config["labels_dir"]
    args.store_mode = config["store_mode"]

    args.task = config["task"]
    args.batch_size = config["batch_size"]
    args.clip_len = config["clip_len"]
    args.dataset = config["dataset"]
    args.epoch_num_frames = config["epoch_num_frames"]
    args.feature_arch = config["feature_arch"]
    args.learning_rate = config["learning_rate"]
    args.num_classes = config["num_classes"]
    args.num_epochs = config["num_epochs"]
    args.warm_up_epochs = config["warm_up_epochs"]
    args.only_test = config["only_test"]
    args.device = config["device"]
    args.num_workers = config["num_workers"]

    # Optional flags for weighted BCE
    args.use_weighted_bce = config.get("use_weighted_bce", False)
    args.pos_weight_clip = float(config.get("pos_weight_clip", 20.0))
    args.pos_weight_eps = float(config.get("pos_weight_eps", 1.0))

    #Optional flags for temporal handler
    args.temporal_handler = config.get("temporal_handler", None)

    #Optional flags for overfitting problems
    args.freeze_backbone = config.get("freeze_backbone", False)
    args.unfreeze_num = config.get("freeze_backbone", 0)
    args.use_focal_loss = config.get("use_focal_loss", False)

    #Optional flags for temporal feature extractors

    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print(
        "Using Linear Warmup ({}) + Cosine Annealing LR ({})".format(
            args.warm_up_epochs, cosine_epochs
        )
    )
    return args.num_epochs, ChainedScheduler(
        [
            LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=args.warm_up_epochs * num_steps_per_epoch,
            ),
            CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs),
        ]
    )


def compute_pos_weight_from_stored_train_labels(args):
    """
    Computes class-wise pos_weight for BCEWithLogitsLoss from stored train labels.

    We use the stored split:
      <args.store_dir>/LEN{clip_len}SPLITtrain/labels.pkl

    labels.pkl is a list with length = number of train clips.
    Each element is a list of annotation dicts. Each dict contains at least "label".
    Labels are 1-based class ids.
    """
    labels_pkl = os.path.join(
        args.store_dir,
        f"LEN{args.clip_len}SPLITtrain",
        "labels.pkl",
    )

    if not os.path.exists(labels_pkl):
        raise FileNotFoundError(
            f"Could not find stored train labels at: {labels_pkl}\n"
            "Run store mode first."
        )

    with open(labels_pkl, "rb") as f:
        labels_store = pickle.load(f)

    num_clips = len(labels_store)
    if num_clips == 0:
        raise ValueError(
            f"Stored train labels are empty at: {labels_pkl}\n"
            "Rebuild the train split in store mode."
        )

    pos_counts = np.zeros(args.num_classes, dtype=np.float64)

    for clip_labels in labels_store:
        present = np.zeros(args.num_classes, dtype=np.float64)

        for item in clip_labels:
            label = int(item["label"])
            if 1 <= label <= args.num_classes:
                present[label - 1] = 1.0

        pos_counts += present

    neg_counts = num_clips - pos_counts

    # Standard heuristic:
    # pos_weight[c] = N_neg[c] / N_pos[c]
    #
    # We protect against division by zero with eps
    # and clip extremely large weights to keep training stable.
    pos_weight = neg_counts / np.maximum(pos_counts, args.pos_weight_eps)
    pos_weight = np.clip(pos_weight, 1.0, args.pos_weight_clip)

    print("=" * 60)
    print("WEIGHTED BCE")
    print("=" * 60)
    print(f"Using weighted BCE: {args.use_weighted_bce}")
    print(f"Train clips used for weights: {num_clips}")
    print(f"pos_weight_clip: {args.pos_weight_clip}")
    print(f"pos_weight_eps: {args.pos_weight_eps}")
    print("Positive counts per class:")
    print(pos_counts.astype(int).tolist())
    print("Computed pos_weight:")
    print(pos_weight.tolist())

    return torch.tensor(pos_weight, dtype=torch.float32)


def main(args):
    print("Setting seed to: ", args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = "config/" + args.model + ".json"
    config = load_json(config_path)
    args = update_args(args, config)

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == "store":
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print("Datasets have been loaded from previous versions correctly!")

    epoch = 0

    def worker_init_fn(worker_id):
        random.seed(worker_id + epoch * 100)

    train_loader = DataLoader(
        train_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn,
    )

    pos_weight = None
    if args.use_weighted_bce:
        pos_weight = compute_pos_weight_from_stored_train_labels(args)

    model = None

    if args.feature_arch.startswith(("rny002", "rny004", "rny008")):
        model = Model(args=args, pos_weight=pos_weight)
    else:
        model = ModelTemporal(args=args, pos_weight=pos_weight)

    optimizer, scaler = model.get_optimizer({"lr": args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = float("inf")

        print("START TRAINING EPOCHS")
        for epoch in range(epoch, num_epochs):
            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler
            )
            val_loss = model.epoch(val_loader)

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True

            print(
                "[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}".format(
                    epoch, train_loss, val_loss
                )
            )
            if better:
                print("New best mAP epoch!")

            losses.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_loss,
                }
            )

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, "loss.json"), losses, pretty=True)

            if better:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "checkpoint_best.pt"),
                )

    print("START INFERENCE")
    model.load(torch.load(os.path.join(ckpt_dir, "checkpoint_best.pt")))

    dummy_input = torch.randn(4, 50, 3, 398, 224).to(args.device)

    macs, params = profile(model._model, inputs=(dummy_input, ))

    print("MACs:", macs)
    print("Params:", params)

    ap_score = evaluate(model, test_data)

    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i] * 100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    avg_table = [["Average", f"{np.mean(ap_score) * 100:.2f}"]]
    headers = ["", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    print("CORRECTLY FINISHED TRAINING AND INFERENCE")


if __name__ == "__main__":
    main(get_args())