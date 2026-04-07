#!/usr/bin/env python3
"""
python stratify_labels.py \
  --splits_dir /ghome/group03/shinto/c6_soccer/SoccerNet/SN-BAS-2025_savedata/splits \
  --clip_len 50 \
  --class_file /ghome/group03/shinto/c6_soccer/CVMasterActionRecognitionSpotting-main/data/soccernetball/class.txt \
  --plots_dir /ghome/group03/shinto/c6_soccer/SoccerNet/SN-BAS-2025_savedata/stratification_plots \
  --strategy median \
  --min_target 8 \
  --background_multiplier 2.0 \
  --seed 1

Stratify / rebalance the TRAIN split of the CVMasterActionRecognitionSpotting baseline.

What it does:
1) Loads the stored training clips from:
   <splits_dir>/LEN<clip_len>SPLITtrain/frame_paths.pkl
   <splits_dir>/LEN<clip_len>SPLITtrain/labels.pkl

2) Computes class distribution BEFORE stratification.
3) Assigns each clip a "primary class":
   - background (0) if the clip has no events
   - otherwise, the rarest class present in that clip
4) Caps the number of clips per positive primary class using a target count
   based on the MEDIAN (default) of the positive class counts.
5) Caps the number of background clips using background_multiplier * target.
6) Saves:
   - two plots: before / after
   - a JSON summary
   - a backup of the original train split
   - the new stratified train split (overwriting frame_paths.pkl and labels.pkl)

Notes:
- This only changes TRAIN. Validation and test remain untouched.
- This is an approximate balancing because clips may contain more than one class.
- Run this AFTER the baseline "store" phase and BEFORE training with "load".
"""

import argparse
import json
import os
import pickle
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratify train clips for the SoccerNet baseline.")
    parser.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="Directory containing LEN<clip_len>SPLITtrain, LEN<clip_len>SPLITval, LEN<clip_len>SPLITtest."
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=50,
        help="Clip length used by the baseline. Default: 50"
    )
    parser.add_argument(
        "--class_file",
        type=str,
        required=True,
        help="Path to data/soccernetball/class.txt"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        required=True,
        help="Directory where before/after plots and summary JSON will be saved."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="median",
        choices=["median", "mean", "quantile"],
        help="How to compute the target cap for positive classes."
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Quantile to use when --strategy quantile. Default: 0.5"
    )
    parser.add_argument(
        "--min_target",
        type=int,
        default=8,
        help="Minimum cap for positive classes. Default: 8"
    )
    parser.add_argument(
        "--background_multiplier",
        type=float,
        default=2.0,
        help="Background cap = background_multiplier * target. Default: 2.0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, compute plots and summary but do not overwrite the training split."
    )
    return parser.parse_args()


def load_class_names(class_file: str) -> List[str]:
    with open(class_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError(f"Empty class file: {class_file}")

    # In this repo class.txt is a single line with space-separated class names.
    # We still support multi-line just in case.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0].split()
    return lines


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def labels_to_presence(clip_labels: List[Dict], num_classes: int) -> np.ndarray:
    """
    Convert one clip's labels (list of dicts) into a multi-hot vector of shape [num_classes].
    Labels in the baseline start at 1.
    """
    presence = np.zeros(num_classes, dtype=np.int64)
    for item in clip_labels:
        label = int(item["label"])
        if 1 <= label <= num_classes:
            presence[label - 1] = 1
    return presence


def compute_distribution(labels_store: List[List[Dict]], num_classes: int) -> Tuple[int, np.ndarray]:
    """
    Returns:
      background_count: number of clips with no events
      class_counts: number of clips containing each class at least once
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    background_count = 0

    for clip_labels in labels_store:
        presence = labels_to_presence(clip_labels, num_classes)
        if presence.sum() == 0:
            background_count += 1
        class_counts += presence

    return background_count, class_counts


def choose_primary_class(clip_labels: List[Dict], global_class_counts: np.ndarray) -> int:
    """
    Primary-class assignment for approximate stratification:
    - 0 for background clips
    - otherwise, choose the rarest class present in the clip
    """
    unique_labels = sorted({
        int(item["label"])
        for item in clip_labels
        if 1 <= int(item["label"]) <= len(global_class_counts)
    })

    if not unique_labels:
        return 0

    # Choose rarest class in the clip; tie-break by smaller label id.
    return min(unique_labels, key=lambda lbl: (global_class_counts[lbl - 1], lbl))


def compute_primary_assignments(labels_store: List[List[Dict]], num_classes: int) -> Tuple[List[int], Dict[int, int]]:
    """
    Returns:
      assignments: one primary class per clip (0 = background)
      primary_counts: count of clips assigned to each primary class
    """
    _, global_class_counts = compute_distribution(labels_store, num_classes)

    assignments: List[int] = []
    primary_counts: Dict[int, int] = defaultdict(int)

    for clip_labels in labels_store:
        primary = choose_primary_class(clip_labels, global_class_counts)
        assignments.append(primary)
        primary_counts[primary] += 1

    for cls_id in range(num_classes + 1):
        primary_counts.setdefault(cls_id, 0)

    return assignments, dict(primary_counts)


def compute_target(primary_counts: Dict[int, int], strategy: str, quantile: float, min_target: int) -> int:
    positive_counts = np.array(
        [count for cls_id, count in primary_counts.items() if cls_id != 0 and count > 0],
        dtype=np.float64
    )

    if len(positive_counts) == 0:
        raise ValueError("No positive clips were found in the training split.")

    if strategy == "median":
        target = float(np.median(positive_counts))
    elif strategy == "mean":
        target = float(np.mean(positive_counts))
    elif strategy == "quantile":
        target = float(np.quantile(positive_counts, quantile))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    target = max(int(round(target)), int(min_target))
    return target


def build_stratified_indices(
    labels_store: List[List[Dict]],
    num_classes: int,
    strategy: str,
    quantile: float,
    min_target: int,
    background_multiplier: float,
    seed: int
) -> Tuple[List[int], Dict]:
    rng = random.Random(seed)

    assignments, primary_counts_before = compute_primary_assignments(labels_store, num_classes)
    target = compute_target(primary_counts_before, strategy, quantile, min_target)

    by_primary: Dict[int, List[int]] = defaultdict(list)
    for idx, primary in enumerate(assignments):
        by_primary[primary].append(idx)

    selected_indices: List[int] = []
    kept_per_primary: Dict[int, int] = {}

    # Positive classes
    for cls_id in range(1, num_classes + 1):
        idxs = by_primary.get(cls_id, [])
        if len(idxs) <= target:
            kept = idxs[:]
        else:
            kept = rng.sample(idxs, target)
        selected_indices.extend(kept)
        kept_per_primary[cls_id] = len(kept)

    # Background
    bg_idxs = by_primary.get(0, [])
    bg_target = max(int(round(background_multiplier * target)), target)
    bg_target = min(bg_target, len(bg_idxs))

    if len(bg_idxs) <= bg_target:
        kept_bg = bg_idxs[:]
    else:
        kept_bg = rng.sample(bg_idxs, bg_target)

    selected_indices.extend(kept_bg)
    kept_per_primary[0] = len(kept_bg)

    selected_indices = sorted(selected_indices)

    stats = {
        "target_per_positive_primary_class": target,
        "background_target": bg_target,
        "primary_counts_before": primary_counts_before,
        "primary_counts_after": kept_per_primary,
        "num_selected_clips": len(selected_indices),
        "num_original_clips": len(labels_store),
    }

    return selected_indices, stats


def make_distribution_plot(
    background_count: int,
    class_counts: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: str
) -> None:
    labels = ["BACKGROUND"] + class_names
    values = [int(background_count)] + [int(x) for x in class_counts.tolist()]

    fig_width = max(12, len(labels) * 0.9)
    plt.figure(figsize=(fig_width, 6))
    bars = plt.bar(range(len(labels)), values)

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Number of clips")
    plt.title(title)

    ymax = max(values) if values else 1
    offset = max(1, int(0.01 * ymax))

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if value > 999 else 0
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def ensure_backup(train_split_dir: str) -> str:
    backup_dir = train_split_dir + "_backup_original"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(os.path.join(train_split_dir, "frame_paths.pkl"), os.path.join(backup_dir, "frame_paths.pkl"))
        shutil.copy2(os.path.join(train_split_dir, "labels.pkl"), os.path.join(backup_dir, "labels.pkl"))
    return backup_dir


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.plots_dir, exist_ok=True)

    train_split_dir = os.path.join(args.splits_dir, f"LEN{args.clip_len}SPLITtrain")
    frame_paths_pkl = os.path.join(train_split_dir, "frame_paths.pkl")
    labels_pkl = os.path.join(train_split_dir, "labels.pkl")

    if not os.path.exists(frame_paths_pkl):
        raise FileNotFoundError(f"Missing file: {frame_paths_pkl}")
    if not os.path.exists(labels_pkl):
        raise FileNotFoundError(f"Missing file: {labels_pkl}")

    class_names = load_class_names(args.class_file)
    num_classes = len(class_names)

    frame_paths = load_pickle(frame_paths_pkl)
    labels_store = load_pickle(labels_pkl)

    if len(frame_paths) != len(labels_store):
        raise ValueError(
            f"Mismatch between frame_paths ({len(frame_paths)}) and labels ({len(labels_store)})"
        )

    if len(frame_paths) == 0:
        raise ValueError("The training split is empty. Run baseline store mode correctly before stratifying.")

    bg_before, class_counts_before = compute_distribution(labels_store, num_classes)

    before_plot = os.path.join(args.plots_dir, "train_distribution_before.png")
    make_distribution_plot(
        bg_before,
        class_counts_before,
        class_names,
        "Train clip distribution BEFORE stratification",
        before_plot
    )

    selected_indices, strat_stats = build_stratified_indices(
        labels_store=labels_store,
        num_classes=num_classes,
        strategy=args.strategy,
        quantile=args.quantile,
        min_target=args.min_target,
        background_multiplier=args.background_multiplier,
        seed=args.seed
    )

    new_frame_paths = [frame_paths[i] for i in selected_indices]
    new_labels_store = [labels_store[i] for i in selected_indices]

    bg_after, class_counts_after = compute_distribution(new_labels_store, num_classes)

    after_plot = os.path.join(args.plots_dir, "train_distribution_after.png")
    make_distribution_plot(
        bg_after,
        class_counts_after,
        class_names,
        "Train clip distribution AFTER stratification",
        after_plot
    )

    summary = {
        "class_names": class_names,
        "num_classes": num_classes,
        "before": {
            "num_clips": len(frame_paths),
            "background_count": int(bg_before),
            "class_counts": {class_names[i]: int(class_counts_before[i]) for i in range(num_classes)},
        },
        "after": {
            "num_clips": len(new_frame_paths),
            "background_count": int(bg_after),
            "class_counts": {class_names[i]: int(class_counts_after[i]) for i in range(num_classes)},
        },
        "stratification": strat_stats,
        "plots": {
            "before": before_plot,
            "after": after_plot,
        },
        "settings": {
            "strategy": args.strategy,
            "quantile": args.quantile,
            "min_target": args.min_target,
            "background_multiplier": args.background_multiplier,
            "seed": args.seed,
            "dry_run": args.dry_run,
        },
    }

    summary_path = os.path.join(args.plots_dir, "stratification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("TRAIN STRATIFICATION SUMMARY")
    print("=" * 60)
    print(f"Original train clips: {len(frame_paths)}")
    print(f"Selected train clips: {len(new_frame_paths)}")
    print(f"Target per positive primary class: {strat_stats['target_per_positive_primary_class']}")
    print(f"Background target: {strat_stats['background_target']}")
    print(f"Before plot: {before_plot}")
    print(f"After plot:  {after_plot}")
    print(f"Summary:     {summary_path}")

    print("\nCounts BEFORE (actual clip distribution):")
    print(f"  BACKGROUND: {int(bg_before)}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int(class_counts_before[i])}")

    print("\nCounts AFTER (actual clip distribution):")
    print(f"  BACKGROUND: {int(bg_after)}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int(class_counts_after[i])}")

    if args.dry_run:
        print("\nDry run enabled: the train split was NOT overwritten.")
        return

    backup_dir = ensure_backup(train_split_dir)
    save_pickle(frame_paths_pkl, new_frame_paths)
    save_pickle(labels_pkl, new_labels_store)

    selected_idx_path = os.path.join(args.plots_dir, "selected_train_indices.json")
    with open(selected_idx_path, "w", encoding="utf-8") as f:
        json.dump(selected_indices, f)

    print("\nOriginal train split backed up to:")
    print(f"  {backup_dir}")
    print("Stratified train split written in-place to:")
    print(f"  {train_split_dir}")
    print("Done.")


if __name__ == "__main__":
    main()