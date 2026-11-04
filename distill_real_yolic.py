"""
Distill Author's Real Pretrained ResNet50 YOLIC into Edge TinyML Model (ShuffleNetV2 + MHSA + Grouped Head)
Transfers real-world outdoor road hazard representations from the paper author's ONNX model.
"""

import os
import sys
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
from datasets.synthetic_generator import SyntheticSceneGenerator

class DistillationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (image_np_bgr, target_matrix_104_12)
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

def main():
    torch.set_num_threads(2)
    device = torch.device("cpu")
    print("Initializing distillation harness...")

    onnx_path = os.path.join(PROJECT_ROOT, "weights", "Config3_resnet50_rgb.onnx")
    assert os.path.exists(onnx_path), f"Teacher model not found at {onnx_path}"
    teacher = ort.InferenceSession(onnx_path)
    in_name = teacher.get_inputs()[0].name
    out_name = teacher.get_outputs()[0].name

    img_dir = os.path.join(PROJECT_ROOT, "data", "real_author_dataset", "RGB")
    real_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"Loaded {len(real_files)} real author road images.")

    samples = []

    # 1. Real author images labeled by teacher ONNX
    for fpath in real_files:
        img = cv2.imread(fpath)
        # Precompute teacher output
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        out = teacher.run([out_name], {in_name: tensor})[0].reshape(104, 12)
        probs = 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))

        # Add original and flipped/jittered variants
        samples.append((img, probs))
        samples.append((cv2.flip(img, 1), probs))  # Augmented flip

    # 2. Add synthetic clear road and obstacle samples for robust balance
    gen = SyntheticSceneGenerator(num_cois=104, num_classes=11)
    for _ in range(30):
        # Clear road
        img, tgt, _ = gen.generate_frame(num_hazards=0)
        samples.append((img, tgt))
    for _ in range(40):
        # Hazard road
        img, tgt, _ = gen.generate_frame(num_hazards=np.random.randint(1, 4))
        samples.append((img, tgt))

    print(f"Total training dataset size: {len(samples)} augmented samples.")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = DistillationDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Initialize UnifiedYOLIC student model
    student = UnifiedYOLIC(
        num_cois=104,
        num_classes=11,
        backbone_type="shufflenet_v2",
        cell_dim=128,
        attention_heads=4,
        use_context_module=True,
        use_grouped_head=True,
        pretrained_backbone=False
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Beginning student model fine-tuning...")
    student.train()
    for epoch in range(1, 21):
        total_loss = 0.0
        for imgs, tgts in loader:
            optimizer.zero_grad()
            preds = student(imgs)
            loss = criterion(preds, tgts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/20 - BCE Loss: {avg_loss:.4f}")

    # 4. Save checkpoint
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "outdoor_104coi")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    torch.save(student.state_dict(), ckpt_path)
    print(f"Student model successfully trained and saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
