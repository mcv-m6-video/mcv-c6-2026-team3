#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model
from model.model_spotting_mod import ModelMod
from thop import profile


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    # Optional temporal head configuration
    args.use_temporal_head = config.get('use_temporal_head', False)
    args.temporal_hidden_dim = config.get('temporal_hidden_dim', None)
    args.temporal_kernel_size = config.get('temporal_kernel_size', 3)
    args.temporal_dilations = config.get('temporal_dilations', [1, 2, 4])
    args.temporal_dropout = config.get('temporal_dropout', 0.2)

    #Augmentation params
    args.clip_aug = config.get("clip_aug", False)

    #Patience
    args.patience = config.get("patience", 5)

    #Model params
    args.use_delta = config.get("use_delta", False)
    args.use_bottleneck = config.get("use_bottleneck", False)
    if args.use_bottleneck:
        print("USING BOTTLENECK ON DELTA")

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])

def compute_ap10(classes, ap_score, exclude=("FREE KICK", "GOAL")):
    exclude = set(exclude)
    ap10_scores = [
        ap_score[i]
        for i, class_name in enumerate(classes.keys())
        if class_name not in exclude
    ]
    ap10 = float(np.mean(ap10_scores))

    return ap10


def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data, val_video_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model

    model = None
    if args.clip_aug:
        model = ModelMod(args=args)
    else: 
        model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = 0
        epoch = 0
        epochs_no_improve = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)

            better = False

            #Added AP model selector
            ap12, ap_score, val_loss = evaluate(model, val_video_data, batch_size=args.batch_size, nms_window=5)
            ap10 = compute_ap10(classes, ap_score)

            better = False
            if ap10 > best_criterion:
                better = True
                best_criterion = ap10
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            #Printing info epoch
            print(f'[Epoch {epoch}] Train loss: {train_loss:0.5f} Val loss: {val_loss:0.5f} AP10: {ap10} AP12: {ap12}')
            
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss, 'ap10' : ap10, 'ap12' : ap12
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )

            if args.patience is not None and epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    dummy_input = torch.randn(4, 50, 3, 398, 224).to(args.device)

    macs, _ = profile(model._model, inputs=(dummy_input, ))

    print(f"MACs : {macs}")

    # Evaluation on test split
    ap12, ap_score, _ = evaluate(model, test_data, nms_window = 5)
    ap10 = compute_ap10(classes, ap_score)

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Report average results in table
    avg_table = [["Mean", f"{ap12*100:.2f}"]]
    headers = ["", "Average Precision 12"]

    print(tabulate(avg_table, headers, tablefmt="grid"))

    # Report average results in table
    avg_table = [["Mean", f"{ap10*100:.2f}"]]
    headers = ["", "Average Precision 10"]

    print(tabulate(avg_table, headers, tablefmt="grid"))
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')


if __name__ == '__main__':
    main(get_args())