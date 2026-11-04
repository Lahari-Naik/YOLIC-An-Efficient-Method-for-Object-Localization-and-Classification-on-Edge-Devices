"""
PyTorch Dataset and DataLoader for YOLIC Edge Vision
Supports:
1. Synthetic on-the-fly generation.
2. Official Author Dataset from Kaggle (sukai3316/outdoor-hazard-detection-dataset):
   - images/ (*.jpg, *.png)
   - yoliclabel/ (*.txt with 104x12 binary matrix annotations)
"""
from typing import Callable, Dict, List, Optional, Tuple
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

from .synthetic_generator import SyntheticSceneGenerator, get_coi_polygons


def get_horizontal_flip_indices_104() -> List[int]:
    """Computes index mapping when an image with 104 CoIs is horizontally flipped."""
    flip_map = list(range(104))
    # 96 road cells: 12 rows of 8 columns
    for r in range(12):
        row_start = r * 8
        for c in range(8):
            orig_idx = row_start + c
            flipped_idx = row_start + (7 - c)
            flip_map[orig_idx] = flipped_idx

    # 8 top sign cells: indices 96 to 103
    for c in range(8):
        orig_idx = 96 + c
        flipped_idx = 96 + (7 - c)
        flip_map[orig_idx] = flipped_idx

    return flip_map


class SyntheticYOLICDataset(Dataset):
    """
    On-the-fly synthetic dataset generating realistic road scenes and CoI labels.
    """
    def __init__(
        self,
        length: int = 500,
        num_cois: int = 104,
        num_classes: int = 11,
        is_train: bool = True
    ):
        super().__init__()
        self.length = length
        self.num_cois = num_cois
        self.num_classes = num_classes
        self.is_train = is_train
        self.generator = SyntheticSceneGenerator(num_cois=num_cois, num_classes=num_classes)
        self.flip_indices = get_horizontal_flip_indices_104() if num_cois == 104 else list(range(num_cois))

        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np, target_np, _ = self.generator.generate_frame()
        img_pil = Image.fromarray(img_np)
        target = torch.from_numpy(target_np).float()

        if self.is_train:
            if random.random() > 0.5:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                target = target[self.flip_indices]
            img_pil = self.color_jitter(img_pil)

        img_tensor = T.functional.to_tensor(img_pil)
        img_tensor = self.normalize(img_tensor)
        return img_tensor, target


class RealYOLICDataset(Dataset):
    """
    Dataset loader for official author datasets (e.g. from Kaggle sukai3316/outdoor-hazard-detection-dataset).
    Supports:
    - image_dir: folder of RGB images (e.g. 'data/images')
    - label_dir: folder of .txt label files matching image basenames (e.g. 'data/yoliclabel')
    """
    def __init__(
        self,
        image_dir: str = "data/images",
        label_dir: Optional[str] = "data/yoliclabel",
        label_file: Optional[str] = None,
        num_cois: int = 104,
        num_classes: int = 11,
        is_train: bool = True
    ):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.is_train = is_train
        self.num_cois = num_cois
        self.num_classes = num_classes
        self.total_classes = num_classes + 1
        self.flip_indices = get_horizontal_flip_indices_104() if num_cois == 104 else list(range(num_cois))

        self.image_paths = []
        if os.path.isdir(image_dir):
            for fname in sorted(os.listdir(image_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(image_dir, fname))

        self.json_labels = {}
        if label_file and os.path.exists(label_file):
            import json
            with open(label_file, "r") as f:
                self.json_labels = json.load(f)

        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize((224, 224))

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 1. Check for author's .txt annotation (yoliclabel/<name>.txt)
        txt_path = os.path.join(self.label_dir, f"{base_name}.txt") if self.label_dir else None
        if txt_path and os.path.exists(txt_path):
            try:
                raw_arr = np.loadtxt(txt_path, dtype=np.float32)
                # Reshape to (num_cois, total_classes)
                target = torch.from_numpy(raw_arr).view(self.num_cois, self.total_classes)
            except Exception:
                target = torch.zeros((self.num_cois, self.total_classes), dtype=torch.float32)
                target[:, self.num_classes] = 1.0

        elif base_name in self.json_labels:
            target = torch.tensor(self.json_labels[base_name], dtype=torch.float32)
        else:
            # Default fallback: clear road
            target = torch.zeros((self.num_cois, self.total_classes), dtype=torch.float32)
            target[:, self.num_classes] = 1.0

        if self.is_train:
            if random.random() > 0.5:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                target = target[self.flip_indices]
            img_pil = self.color_jitter(img_pil)

        img_tensor = T.functional.to_tensor(img_pil)
        img_tensor = self.normalize(img_tensor)

        return img_tensor, target


def get_dataloader(
    config: dict,
    split: str = "train",
    batch_size: int = 32,
    synthetic: bool = True,
    num_samples: int = 600,
    shuffle: bool = True
) -> DataLoader:
    """Factory to get standard PyTorch DataLoader."""
    num_cois = config.get("dataset", {}).get("num_cois", 104)
    num_classes = config.get("dataset", {}).get("num_classes", 11)

    if synthetic:
        dataset = SyntheticYOLICDataset(
            length=num_samples if split == "train" else max(100, num_samples // 4),
            num_cois=num_cois,
            num_classes=num_classes,
            is_train=(split == "train")
        )
    else:
        img_dir = config.get("dataset", {}).get(f"{split}_img_dir", "data/images")
        lbl_dir = config.get("dataset", {}).get(f"{split}_label_dir", "data/yoliclabel")
        lbl_file = config.get("dataset", {}).get(f"{split}_label_file", None)
        dataset = RealYOLICDataset(
            image_dir=img_dir,
            label_dir=lbl_dir,
            label_file=lbl_file,
            num_cois=num_cois,
            num_classes=num_classes,
            is_train=(split == "train")
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (split == "train"),
        num_workers=0,
        pin_memory=False
    )
    return loader
