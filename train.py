"""
Training Pipeline for Extended YOLIC
Loss: Binary Cross-Entropy with Logits
Optimizer: Adam (lr=0.001, weight_decay=1e-4)
Scheduler: MultiStepLR at epochs [100, 125] with gamma=0.1
"""
from typing import Optional
import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from models.unified_yolic import UnifiedYOLIC, build_yolic_from_config
from datasets.coi_dataset import get_dataloader


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Computes Road-vs-Risk binary F1 and multi-label hazard F1.
    """
    probs = torch.sigmoid(logits)
    bg_idx = logits.size(-1) - 1

    # Binary Road vs Risk
    pred_road = (probs[:, :, bg_idx] >= threshold).float()
    true_road = targets[:, :, bg_idx]

    tp_road = ((pred_road == 1) & (true_road == 1)).sum().item()
    fp_road = ((pred_road == 1) & (true_road == 0)).sum().item()
    fn_road = ((pred_road == 0) & (true_road == 1)).sum().item()

    prec_road = tp_road / (tp_road + fp_road + 1e-8)
    rec_road = tp_road / (tp_road + fn_road + 1e-8)
    f1_road = 2 * (prec_road * rec_road) / (prec_road + rec_road + 1e-8)

    # Multi-label Hazard F1 (classes 0 to M-1)
    pred_hazards = (probs[:, :, :bg_idx] >= threshold).float()
    true_hazards = targets[:, :, :bg_idx]

    tp_h = ((pred_hazards == 1) & (true_hazards == 1)).sum().item()
    fp_h = ((pred_hazards == 1) & (true_hazards == 0)).sum().item()
    fn_h = ((pred_hazards == 0) & (true_hazards == 1)).sum().item()

    prec_h = tp_h / (tp_h + fp_h + 1e-8)
    rec_h = tp_h / (tp_h + fn_h + 1e-8)
    f1_hazard = 2 * (prec_h * rec_h) / (prec_h + rec_h + 1e-8)

    return f1_road, f1_hazard


def train(args):
    # Load configuration
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    else:
        print(f"Warning: Config file {args.config} not found. Using default arguments.")

    epochs = args.epochs if args.epochs is not None else cfg.get("training", {}).get("epochs", 150)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("training", {}).get("batch_size", 32)
    lr = args.lr if args.lr is not None else cfg.get("training", {}).get("learning_rate", 0.001)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    milestones = cfg.get("training", {}).get("lr_milestones", [100, 125])
    gamma = cfg.get("training", {}).get("lr_gamma", 0.1)
    save_dir = cfg.get("training", {}).get("save_dir", "checkpoints/outdoor_104coi")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Using compute device: {device}")

    # Build model
    if cfg:
        model = build_yolic_from_config(cfg, pretrained=not args.no_pretrained).to(device)
    else:
        model = UnifiedYOLIC(
            num_cois=104,
            num_classes=11,
            backbone_type="shufflenet_v2",
            use_context_module=True,
            use_grouped_head=True,
            pretrained_backbone=not args.no_pretrained
        ).to(device)

    # Loss function: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Data loaders
    train_loader = get_dataloader(cfg, split="train", batch_size=batch_size, synthetic=args.synthetic, num_samples=args.samples)
    val_loader = get_dataloader(cfg, split="val", batch_size=batch_size, synthetic=args.synthetic, num_samples=max(50, args.samples // 4))

    print(f"Starting training: {epochs} epochs | Batch size {batch_size} | Synthetic mode: {args.synthetic}")
    print("=" * 70)

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, num_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        val_loss = val_loss / max(1, val_batches)
        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        f1_road, f1_hazard = compute_metrics(cat_logits, cat_targets)

        dt = time.time() - t0
        print(f"Epoch [{epoch:03d}/{epochs:03d}] ({dt:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Road F1: {f1_road:.3f} | Hazard F1: {f1_hazard:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpointing
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "f1_road": f1_road,
            "f1_hazard": f1_hazard,
            "val_loss": val_loss,
            "config": cfg
        }
        torch.save(checkpoint, os.path.join(save_dir, "last_model.pth"))

        if f1_hazard > best_val_f1:
            best_val_f1 = f1_hazard
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"  --> Saved new best checkpoint (Hazard F1: {best_val_f1:.4f})")

    print("=" * 70)
    print(f"Training Complete! Best Hazard F1: {best_val_f1:.4f}")
    print(f"Checkpoints saved in '{save_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended YOLIC Training Script")
    parser.add_argument("--config", type=str, default="configs/outdoor_104coi.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples per epoch")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset generator")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args)
