import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.samplers import MPerClassSampler
import argparse
import os
import random
import numpy as np


def seed_everything(seed):
    """Set all seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"--- End Epoch {epoch} | Average Loss: {avg_loss:.4f} ---")
    return avg_loss


def evaluate(model, loss_func, device, val_loader, epoch):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)
            loss = loss_func(embeddings, labels)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f"--- End Epoch {epoch} | Validation Loss: {avg_val_loss:.4f} ---")
    return avg_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Siamese ResNet")
    parser.add_argument("--data_root", required=True, help="Path to gt_crops directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"Using seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets with different transforms for train/val
    full_train_dataset = datasets.ImageFolder(root=args.data_root, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(root=args.data_root, transform=val_transform)

    print(f"Loaded {len(full_train_dataset)} images from {len(full_train_dataset.classes)} tracks")

    # Train/val split keeping same indices for both datasets
    num_samples = len(full_train_dataset)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(args.seed)).tolist()

    val_size = max(1, int(num_samples * args.val_split)) if args.val_split > 0 else 0
    train_size = num_samples - val_size

    if val_size > 0:
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_val_dataset, val_idx)
    else:
        train_dataset = full_train_dataset
        val_dataset = None

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_labels = [full_train_dataset.targets[i] for i in train_idx] if val_size > 0 else full_train_dataset.targets
    sampler = MPerClassSampler(train_labels, m=4, length_before_new_iter=len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

    # Load pretrained ResNet18 as feature extractor
    resnet_base = models.resnet18(weights=None)
    weights_path = os.path.join(script_dir, "resnet18-f37072fd.pth")
    resnet_base.load_state_dict(torch.load(weights_path, map_location=device))

    extractor = nn.Sequential(*(list(resnet_base.children())[:-1]))
    model = nn.Sequential(
        extractor,
        nn.Flatten(),
        nn.BatchNorm1d(512)
    ).to(device)

    # Optimizer, scheduler and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    distance = distances.CosineSimilarity()
    loss_func = losses.TripletMarginLoss(distance=distance, margin=args.margin)
    mining_func = miners.TripletMarginMiner(margin=args.margin, distance=distance, type_of_triplets="semihard")

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
        scheduler.step()

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, loss_func, device, val_loader, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_checkpoint_path = os.path.join(output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_to_idx': full_train_dataset.class_to_idx,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print(f"New best model saved to: {best_checkpoint_path} (val_loss={best_val_loss:.4f})")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs:
            checkpoint_name = os.path.join(output_dir, f"resnet18_reid_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': full_train_dataset.class_to_idx,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")

    if best_epoch != -1:
        print(f"Best validation model from epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    main()