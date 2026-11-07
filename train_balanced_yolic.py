"""
Balanced Training Pipeline for Unified YOLIC (ShuffleNetV2 TinyML)
Uses ImageNet Pretrained ShuffleNetV2 Backbone + Multi-Domain Dataset
Ensures active hazard sensitivity while preventing hallucinations on clean backgrounds.
"""

import os
import sys
import math
import random
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unified_yolic import UnifiedYOLIC
from datasets.synthetic_generator import SyntheticSceneGenerator, get_coi_polygons

CLASS_NAMES = [
    "Bump", "Column", "Dent", "Fence", "Creature",
    "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone", "TrafficSign"
]
NUM_COIS = 104
NUM_CLASSES = 11
TOTAL_CLASSES = 12
BG_INDEX = 11


class BalancedPerceptionDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bgr, target = self.samples[idx]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = T.functional.to_pil_image(img_rgb)
        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = T.functional.to_tensor(pil_img)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return tensor, target_tensor


def create_clean_background_target() -> np.ndarray:
    tgt = np.zeros((NUM_COIS, TOTAL_CLASSES), dtype=np.float32)
    tgt[:, BG_INDEX] = 1.0
    return tgt


def generate_indoor_room_sample() -> Tuple[np.ndarray, np.ndarray]:
    H, W = 224, 224
    frame_type = random.choice(["wood_floor", "carpet", "plain_wall", "desk_surface", "tile_floor"])
    img = np.zeros((H, W, 3), dtype=np.uint8)

    if frame_type == "wood_floor":
        base_color = np.array([random.randint(130, 170), random.randint(160, 200), random.randint(190, 230)], dtype=np.uint8)
        img[:] = base_color
        for y in range(0, H, random.randint(18, 28)):
            cv2.line(img, (0, y), (W, y), (base_color * 0.75).astype(np.uint8).tolist(), 1)
    elif frame_type == "carpet":
        base_gray = random.randint(90, 180)
        img[:] = (base_gray, base_gray + random.randint(-5, 5), base_gray + random.randint(-5, 5))
        noise = np.random.randint(-15, 15, (H, W, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif frame_type == "plain_wall":
        wall_color = random.randint(190, 245)
        img[:] = (wall_color, wall_color, wall_color)
        cv2.rectangle(img, (0, H - 25), (W, H), (140, 140, 150), -1)
    elif frame_type == "desk_surface":
        desk_c = (random.randint(110, 160), random.randint(130, 180), random.randint(150, 200))
        img[:] = desk_c
        cv2.rectangle(img, (random.randint(40, 80), random.randint(60, 120)), (random.randint(140, 190), random.randint(160, 210)), (40, 40, 45), -1)
    else:
        img[:] = (210, 215, 220)
        grid_s = random.randint(30, 50)
        for x in range(0, W, grid_s):
            cv2.line(img, (x, 0), (x, H), (160, 165, 170), 2)
        for y in range(0, H, grid_s):
            cv2.line(img, (0, y), (W, y), (160, 165, 170), 2)

    return img, create_clean_background_target()


def generate_webcam_person_sample() -> Tuple[np.ndarray, np.ndarray]:
    img, _ = generate_indoor_room_sample()
    H, W = 224, 224
    polygons = get_coi_polygons(NUM_COIS)

    cx = random.randint(80, 144)
    cy = random.randint(100, 150)
    head_r = random.randint(18, 28)

    # Face/head
    skin_color = (random.randint(140, 220), random.randint(160, 200), random.randint(190, 235))
    cv2.circle(img, (cx, cy - head_r - 15), head_r, skin_color, -1)
    # Hair
    cv2.ellipse(img, (cx, cy - head_r - 22), (head_r + 3, head_r // 2 + 2), 0, 180, 360, (30, 25, 20), -1)
    # Torso
    torso_color = (random.randint(30, 220), random.randint(30, 220), random.randint(30, 220))
    cv2.rectangle(img, (cx - head_r - 22, cy - 5), (cx + head_r + 22, H), torso_color, -1)

    target = create_clean_background_target()

    for idx, poly in enumerate(polygons):
        poly_px = (poly * [W, H]).astype(np.int32)
        center_x = int(np.mean(poly_px[:, 0]))
        center_y = int(np.mean(poly_px[:, 1]))

        dist_head = math.hypot(center_x - cx, center_y - (cy - head_r - 15))
        in_torso = (cx - head_r - 24 <= center_x <= cx + head_r + 24) and (cy - 10 <= center_y <= H)

        if dist_head <= head_r or in_torso:
            target[idx, 4] = 1.0  # Creature
            target[idx, BG_INDEX] = 0.0

    return img, target


def assemble_dataset():
    generator = SyntheticSceneGenerator(num_cois=NUM_COIS, num_classes=NUM_CLASSES)
    train_samples = []
    val_samples = []

    # 1. Real author dataset with teacher distillation
    onnx_path = os.path.join(PROJECT_ROOT, "weights", "Config3_resnet50_rgb.onnx")
    real_img_dir = os.path.join(PROJECT_ROOT, "data", "real_author_dataset", "RGB")

    if os.path.exists(onnx_path) and os.path.exists(real_img_dir):
        teacher = ort.InferenceSession(onnx_path)
        in_name = teacher.get_inputs()[0].name
        out_name = teacher.get_outputs()[0].name

        for rf in sorted(os.listdir(real_img_dir)):
            if not rf.endswith(".png"):
                continue
            r_img = cv2.imread(os.path.join(real_img_dir, rf))
            resized = cv2.resize(r_img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            out = teacher.run([out_name], {in_name: tensor})[0].reshape(NUM_COIS, TOTAL_CLASSES)
            probs = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))

            for _ in range(8):
                train_samples.append((r_img, probs))
                train_samples.append((cv2.flip(r_img, 1), probs))
            val_samples.append((r_img, probs))

    # 2. Clear road negative samples
    for _ in range(150):
        img, tgt, _ = generator.generate_frame(num_hazards=0, lane_offset=random.uniform(0.0, 1.0))
        train_samples.append((img, tgt))
    for _ in range(30):
        img, tgt, _ = generator.generate_frame(num_hazards=0)
        val_samples.append((img, tgt))

    # 3. Indoor room negative samples
    for _ in range(100):
        img, tgt = generate_indoor_room_sample()
        train_samples.append((img, tgt))
    for _ in range(20):
        img, tgt = generate_indoor_room_sample()
        val_samples.append((img, tgt))

    # 4. Human / Creature samples
    for _ in range(120):
        img, tgt = generate_webcam_person_sample()
        train_samples.append((img, tgt))
    for _ in range(25):
        img, tgt = generate_webcam_person_sample()
        val_samples.append((img, tgt))

    # 5. Multi-hazard road scenes
    for cls_idx in range(NUM_CLASSES):
        for _ in range(30):
            coi_idx = random.randint(8, 88)
            img, tgt, _ = generator.generate_frame(forced_hazards=[(coi_idx, cls_idx)])
            train_samples.append((img, tgt))
        for _ in range(6):
            coi_idx = random.randint(8, 88)
            img, tgt, _ = generator.generate_frame(forced_hazards=[(coi_idx, cls_idx)])
            val_samples.append((img, tgt))

    # Multi-hazard combinations
    for _ in range(120):
        img, tgt, _ = generator.generate_frame(num_hazards=random.randint(1, 3))
        train_samples.append((img, tgt))
    for _ in range(25):
        img, tgt, _ = generator.generate_frame(num_hazards=2)
        val_samples.append((img, tgt))

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    print(f"Dataset assembled: {len(train_samples)} train, {len(val_samples)} val.")
    return train_samples, val_samples


def train():
    torch.set_num_threads(2)
    train_samples, val_samples = assemble_dataset()

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BalancedPerceptionDataset(train_samples, train_tf), batch_size=32, shuffle=True)
    val_loader = DataLoader(BalancedPerceptionDataset(val_samples, val_tf), batch_size=32, shuffle=False)

    model = UnifiedYOLIC(
        num_cois=NUM_COIS,
        num_classes=NUM_CLASSES,
        backbone_type="shufflenet_v2",
        cell_dim=128,
        attention_heads=4,
        use_context_module=True,
        use_grouped_head=True,
        pretrained_backbone=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "outdoor_104coi")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    best_loss = float("inf")
    print("Starting balanced training for 12 epochs...")

    for epoch in range(1, 13):
        model.train()
        total_train_loss = 0.0
        for imgs, tgts in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = total_train_loss / len(train_samples)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                logits = model(imgs)
                loss = criterion(logits, tgts)
                total_val_loss += loss.item() * imgs.size(0)

        val_loss = total_val_loss / len(val_samples)
        print(f"Epoch {epoch:02d}/12 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model checkpoint to {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
