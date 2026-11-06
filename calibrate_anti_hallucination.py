"""
Fast Hard-Negative Calibration for Zero-Hallucination Edge Vision
Uses batched PyTorch DataLoader to calibrate background logits on clean road and indoor scenes.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms as T
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unified_yolic import UnifiedYOLIC
from datasets.synthetic_generator import SyntheticSceneGenerator

class CalibDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bgr, tgt = self.samples[idx]
        pil = T.functional.to_pil_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return self.transform(pil), torch.tensor(tgt, dtype=torch.float32)

def main():
    torch.set_num_threads(2)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "outdoor_104coi", "best_model.pth")
    model = UnifiedYOLIC(
        num_cois=104,
        num_classes=11,
        backbone_type="shufflenet_v2",
        cell_dim=128,
        attention_heads=4,
        use_context_module=True,
        use_grouped_head=True,
        pretrained_backbone=False
    )
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded existing checkpoint from {ckpt_path}")

    generator = SyntheticSceneGenerator(num_cois=104, num_classes=11)
    samples = []

    # 1. Clean road negative samples
    clean_tgt = np.zeros((104, 12), dtype=np.float32)
    clean_tgt[:, 11] = 1.0

    sc_path = os.path.join(PROJECT_ROOT, "sample_clear.png")
    if os.path.exists(sc_path):
        sc_img = cv2.imread(sc_path)
        for _ in range(40):
            samples.append((sc_img, clean_tgt))
            samples.append((cv2.flip(sc_img, 1), clean_tgt))

    for _ in range(120):
        img, tgt, _ = generator.generate_frame(num_hazards=0)
        samples.append((img, tgt))

    # 2. Indoor / Desk / Floor negative samples (room calibration)
    for _ in range(100):
        H, W = 224, 224
        # Floor / plain texture
        bg_color = np.random.randint(100, 230, (1, 1, 3), dtype=np.uint8)
        img = np.full((H, W, 3), bg_color, dtype=np.uint8)
        noise = np.random.randint(-15, 15, (H, W, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        samples.append((img, clean_tgt))

    # 3. Real Author Positives (teacher distillation)
    onnx_path = os.path.join(PROJECT_ROOT, "weights", "Config3_resnet50_rgb.onnx")
    real_dir = os.path.join(PROJECT_ROOT, "data", "real_author_dataset", "RGB")
    if os.path.exists(onnx_path) and os.path.exists(real_dir):
        sess = ort.InferenceSession(onnx_path)
        for rf in sorted(os.listdir(real_dir)):
            if not rf.endswith(".png"):
                continue
            r_img = cv2.imread(os.path.join(real_dir, rf))
            resized = cv2.resize(r_img, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            out = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: tensor})[0].reshape(104, 12)
            probs = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))
            for _ in range(6):
                samples.append((r_img, probs))

    # 4. Multi-hazard synthetic samples
    for cls_id in range(11):
        for _ in range(25):
            coi_idx = np.random.randint(10, 85)
            img, tgt, _ = generator.generate_frame(forced_hazards=[(coi_idx, cls_id)])
            samples.append((img, tgt))

    np.random.shuffle(samples)
    print(f"Total calibration samples: {len(samples)}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(CalibDataset(samples, transform), batch_size=32, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for b_imgs, b_tgts in loader:
            optimizer.zero_grad()
            preds = model(b_imgs)
            loss = criterion(preds, b_tgts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_imgs)
        print(f"Epoch {epoch:2d}/10 - Calib Loss: {total_loss / len(samples):.4f}", flush=True)

    torch.save(model.state_dict(), ckpt_path)
    print(f"Calibrated weights successfully saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
