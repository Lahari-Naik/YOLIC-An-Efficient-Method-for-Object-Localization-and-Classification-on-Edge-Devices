"""
Synthetic Road Scene & CoI Grid Generator for YOLIC
Generates:
1. Exact CoI polygon geometry (top sign cells + perspective lower road cells).
2. Photorealistic synthetic road images (224x224) with asphalt, lane lines, perspective, and injected hazards.
3. Ground-truth multi-hot binary label matrices Y in {0, 1}^(N x (M + 1)).
4. Continuous video sequence generator with moving hazards for temporal flicker rate benchmarking.
"""
from typing import Dict, List, Optional, Tuple
import math
import random
import cv2
import numpy as np
import torch


def generate_coi_polygons_104() -> List[np.ndarray]:
    """
    Generates exact 104 CoI polygons in normalized [0, 1] x [0, 1] coordinates.
    - 0 to 95 (96 cells): Perspective road cells arranged from vehicle bumper (bottom) to horizon (top).
      Indices 0 to 20: Near-field road cells (closest to vehicle).
      Indices 21 to 95: Mid/Far-field road cells.
      Organized as 12 rows x 8 columns.
      Row 0 is closest to bumper (y ~ 1.0 to 0.93), Row 11 is near horizon (y ~ 0.38 to 0.33).
    - 96 to 103 (8 cells): Top traffic sign cells arranged across the upper band (y ~ 0.05 to 0.25).
    Returns list of 104 polygon arrays of shape (4, 2) in normalized [0, 1] coordinates.
    """
    polygons = []

    # 1. Road cells (96 cells: 12 rows, 8 columns)
    num_rows = 12
    num_cols = 8
    y_bumper = 0.98
    y_horizon = 0.35

    # Road spans wider at the bottom (near camera) and narrows towards the horizon
    x_bottom_left, x_bottom_right = 0.06, 0.94
    x_top_left, x_top_right = 0.34, 0.66

    for r in range(num_rows):
        # Non-linear perspective spacing along Y: cells are taller near bumper, compressed near horizon
        # t in [0, 1] from bumper to horizon
        t0 = (r / num_rows) ** 1.3
        t1 = ((r + 1) / num_rows) ** 1.3

        y0 = y_bumper - t0 * (y_bumper - y_horizon)
        y1 = y_bumper - t1 * (y_bumper - y_horizon)

        # Left and right bounds at y0 and y1
        x_left_0 = x_bottom_left + t0 * (x_top_left - x_bottom_left)
        x_right_0 = x_bottom_right - t0 * (x_bottom_right - x_top_right)
        x_left_1 = x_bottom_left + t1 * (x_top_left - x_bottom_left)
        x_right_1 = x_bottom_right - t1 * (x_bottom_right - x_top_right)

        for c in range(num_cols):
            fc0 = c / num_cols
            fc1 = (c + 1) / num_cols

            pt_bl = [x_left_0 + fc0 * (x_right_0 - x_left_0), y0]
            pt_br = [x_left_0 + fc1 * (x_right_0 - x_left_0), y0]
            pt_tr = [x_left_1 + fc1 * (x_right_1 - x_left_1), y1]
            pt_tl = [x_left_1 + fc0 * (x_right_1 - x_left_1), y1]

            poly = np.array([pt_bl, pt_br, pt_tr, pt_tl], dtype=np.float32)
            polygons.append(poly)

    assert len(polygons) == 96, f"Expected 96 road cells, got {len(polygons)}"

    # 2. Top traffic sign cells (8 cells: 1 row, 8 columns)
    y_sign_top = 0.05
    y_sign_bottom = 0.25
    x_sign_start = 0.05
    x_sign_end = 0.95
    sign_cols = 8

    for c in range(sign_cols):
        x0 = x_sign_start + (c / sign_cols) * (x_sign_end - x_sign_start)
        x1 = x_sign_start + ((c + 1) / sign_cols) * (x_sign_end - x_sign_start)
        # Margin inside cell
        poly = np.array([
            [x0 + 0.005, y_sign_bottom],
            [x1 - 0.005, y_sign_bottom],
            [x1 - 0.005, y_sign_top],
            [x0 + 0.005, y_sign_top]
        ], dtype=np.float32)
        polygons.append(poly)

    assert len(polygons) == 104, f"Expected 104 total CoIs, got {len(polygons)}"
    return polygons


def generate_coi_polygons_256() -> List[np.ndarray]:
    """Generates 256 CoI polygons (240 perspective road cells + 16 top sign cells)."""
    polygons = []
    num_rows = 20
    num_cols = 12
    y_bumper = 0.98
    y_horizon = 0.35
    x_bottom_left, x_bottom_right = 0.05, 0.95
    x_top_left, x_top_right = 0.35, 0.65

    for r in range(num_rows):
        t0 = (r / num_rows) ** 1.3
        t1 = ((r + 1) / num_rows) ** 1.3
        y0 = y_bumper - t0 * (y_bumper - y_horizon)
        y1 = y_bumper - t1 * (y_bumper - y_horizon)
        x_left_0 = x_bottom_left + t0 * (x_top_left - x_bottom_left)
        x_right_0 = x_bottom_right - t0 * (x_bottom_right - x_top_right)
        x_left_1 = x_bottom_left + t1 * (x_top_left - x_bottom_left)
        x_right_1 = x_bottom_right - t1 * (x_bottom_right - x_top_right)

        for c in range(num_cols):
            fc0 = c / num_cols
            fc1 = (c + 1) / num_cols
            pt_bl = [x_left_0 + fc0 * (x_right_0 - x_left_0), y0]
            pt_br = [x_left_0 + fc1 * (x_right_0 - x_left_0), y0]
            pt_tr = [x_left_1 + fc1 * (x_right_1 - x_left_1), y1]
            pt_tl = [x_left_1 + fc0 * (x_right_1 - x_left_1), y1]
            polygons.append(np.array([pt_bl, pt_br, pt_tr, pt_tl], dtype=np.float32))

    # 16 top sign cells (2 rows of 8)
    for r in range(2):
        y0 = 0.24 - r * 0.10
        y1 = 0.14 - r * 0.10
        for c in range(8):
            x0 = 0.05 + (c / 8) * 0.90
            x1 = 0.05 + ((c + 1) / 8) * 0.90
            polygons.append(np.array([[x0 + 0.003, y0], [x1 - 0.003, y0], [x1 - 0.003, y1], [x0 + 0.003, y1]], dtype=np.float32))

    return polygons


def get_coi_polygons(num_cois: int = 104) -> List[np.ndarray]:
    """Returns CoI polygons for specified number of CoIs."""
    if num_cois == 104:
        return generate_coi_polygons_104()
    elif num_cois == 256:
        return generate_coi_polygons_256()
    else:
        # Default fallback
        return generate_coi_polygons_104()


class SyntheticSceneGenerator:
    """
    Renders realistic synthetic road imagery and corresponding ground-truth CoI target matrices.
    """
    CLASS_NAMES = [
        "Pothole", "Speed Bump", "Manhole Cover", "Pedestrian",
        "Bicycle", "Car", "Debris", "Curb", "Stop Sign",
        "Yield Sign", "Warning Sign"
    ]
    BACKGROUND_NAME = "Clear Road"

    def __init__(self, num_cois: int = 104, num_classes: int = 11, image_size: Tuple[int, int] = (224, 224)):
        self.num_cois = num_cois
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # 11 hazards + 1 background
        self.bg_index = num_classes          # index 11
        self.image_size = image_size
        self.polygons = get_coi_polygons(num_cois)

    def generate_frame(
        self,
        num_hazards: Optional[int] = None,
        forced_hazards: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generates a single synthetic road frame.
        Returns:
            image: uint8 RGB image of shape (H, W, 3)
            target: float32 binary multi-hot matrix of shape (num_cois, num_classes + 1)
            hazard_info: list of dicts describing active hazards
        """
        H, W = self.image_size
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Sky & Horizon
        y_horizon_px = int(0.35 * H)
        # Sky gradient (light blue / grey)
        for y in range(y_horizon_px):
            ratio = y / y_horizon_px
            color = (int(180 + 30 * ratio), int(195 + 20 * ratio), int(215 - 10 * ratio))
            img[y, :] = color

        # Sidewalk / Grass sides
        img[y_horizon_px:, :] = (50, 110, 50)  # Green grass borders

        # 2. Asphalt Road Surface (Perspective trapezoid)
        road_pts = np.array([
            [int(0.06 * W), H],
            [int(0.94 * W), H],
            [int(0.66 * W), y_horizon_px],
            [int(0.34 * W), y_horizon_px]
        ], dtype=np.int32)
        cv2.fillPoly(img, [road_pts], (75, 75, 78))

        # Asphalt noise texture
        noise = np.random.randint(-10, 10, (H, W, 3), dtype=np.int16)
        road_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(road_mask, [road_pts], 255)
        blended = np.clip(img.astype(np.int16) + (noise * (road_mask[:, :, None] > 0)), 0, 255).astype(np.uint8)
        img = blended

        # 3. Road Markings (Solid border lines + dashed center lines)
        # Left edge line
        cv2.line(img, (int(0.08 * W), H), (int(0.35 * W), y_horizon_px), (220, 220, 220), 2, cv2.LINE_AA)
        # Right edge line
        cv2.line(img, (int(0.92 * W), H), (int(0.65 * W), y_horizon_px), (220, 220, 220), 2, cv2.LINE_AA)

        # Dashed center line
        for step in range(8):
            t0 = (step / 8) ** 1.3
            t1 = ((step + 0.5) / 8) ** 1.3
            y0 = int(H - t0 * (H - y_horizon_px))
            y1 = int(H - t1 * (H - y_horizon_px))
            x0 = int(0.50 * W)
            x1 = int(0.50 * W)
            cv2.line(img, (x0, y0), (x1, y1), (230, 210, 50), max(1, int(3 * (1 - t0))), cv2.LINE_AA)

        # 4. Target Matrix Initialization: all cells clear road initially
        target = np.zeros((self.num_cois, self.total_classes), dtype=np.float32)
        target[:, self.bg_index] = 1.0  # Clear road bit = 1

        # 5. Inject Hazards
        hazard_list = []
        if forced_hazards is not None:
            chosen_hazards = forced_hazards
        else:
            k = random.randint(1, 4) if num_hazards is None else num_hazards
            chosen_cois = random.sample(range(self.num_cois), min(k, self.num_cois))
            chosen_hazards = []
            for coi_idx in chosen_cois:
                if coi_idx >= 96:
                    # Traffic sign cells (classes 8, 9, 10)
                    cls_id = random.choice([8, 9, 10])
                else:
                    # Road hazard cells (classes 0 to 7)
                    cls_id = random.choice(list(range(8)))
                chosen_hazards.append((coi_idx, cls_id))

        for coi_idx, cls_id in chosen_hazards:
            poly_norm = self.polygons[coi_idx]
            poly_px = (poly_norm * [W, H]).astype(np.int32)
            cx = int(np.mean(poly_px[:, 0]))
            cy = int(np.mean(poly_px[:, 1]))
            bw = int(np.ptp(poly_px[:, 0]))
            bh = int(np.ptp(poly_px[:, 1]))

            # Mark target matrix
            target[coi_idx, cls_id] = 1.0
            target[coi_idx, self.bg_index] = 0.0  # Hazard present -> not clear road

            # Render visual cue on image
            cls_name = self.CLASS_NAMES[cls_id]
            self._render_hazard(img, cls_id, cx, cy, bw, bh, poly_px)
            hazard_list.append({
                "coi_id": coi_idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy)
            })

        return img, target, hazard_list

    def _render_hazard(self, img: np.ndarray, cls_id: int, cx: int, cy: int, bw: int, bh: int, poly_px: np.ndarray):
        """Draws representative visual markers for hazards."""
        radius = max(3, min(bw, bh) // 3)
        if cls_id == 0:  # Pothole
            cv2.ellipse(img, (cx, cy), (radius, max(2, radius // 2)), 0, 0, 360, (25, 25, 28), -1)
            cv2.ellipse(img, (cx, cy), (radius, max(2, radius // 2)), 0, 0, 360, (15, 15, 15), 1)
        elif cls_id == 1:  # Speed Bump
            cv2.ellipse(img, (cx, cy), (max(4, bw // 2 - 2), max(2, bh // 4)), 0, 0, 360, (30, 200, 240), -1)
            cv2.line(img, (cx - bw // 3, cy), (cx + bw // 3, cy), (20, 20, 20), 2)
        elif cls_id == 2:  # Manhole Cover
            cv2.circle(img, (cx, cy), radius, (90, 95, 100), -1)
            cv2.circle(img, (cx, cy), radius, (50, 50, 50), 2)
        elif cls_id == 3:  # Pedestrian
            cv2.circle(img, (cx, max(0, cy - radius)), max(2, radius // 2), (230, 180, 140), -1)
            cv2.rectangle(img, (cx - radius // 2, cy - radius // 2), (cx + radius // 2, cy + radius), (40, 60, 200), -1)
        elif cls_id == 4:  # Bicycle
            cv2.circle(img, (cx - radius // 2, cy), max(2, radius // 2), (20, 150, 220), 2)
            cv2.circle(img, (cx + radius // 2, cy), max(2, radius // 2), (20, 150, 220), 2)
            cv2.line(img, (cx - radius // 2, cy), (cx + radius // 2, cy), (180, 20, 20), 2)
        elif cls_id == 5:  # Car
            cv2.rectangle(img, (cx - bw // 3, cy - bh // 3), (cx + bw // 3, cy + bh // 3), (200, 50, 50), -1)
            cv2.rectangle(img, (cx - bw // 4, cy - bh // 4), (cx + bw // 4, cy), (240, 240, 250), -1)
        elif cls_id == 6:  # Debris
            for _ in range(4):
                dx = random.randint(-radius, radius)
                dy = random.randint(-radius, radius)
                cv2.circle(img, (cx + dx, cy + dy), 2, (139, 69, 19), -1)
        elif cls_id == 7:  # Curb
            cv2.line(img, (cx - bw // 2, cy), (cx + bw // 2, cy), (180, 180, 180), 3)
        elif cls_id == 8:  # Stop Sign (octagonal red)
            pts = np.array([
                [cx - radius, cy - radius // 2], [cx - radius // 2, cy - radius],
                [cx + radius // 2, cy - radius], [cx + radius, cy - radius // 2],
                [cx + radius, cy + radius // 2], [cx + radius // 2, cy + radius],
                [cx - radius // 2, cy + radius], [cx - radius, cy + radius // 2]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (220, 30, 30))
        elif cls_id == 9:  # Yield Sign (triangle)
            pts = np.array([[cx - radius, cy - radius], [cx + radius, cy - radius], [cx, cy + radius]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (240, 240, 240))
            cv2.polylines(img, [pts], True, (220, 30, 30), 2)
        elif cls_id == 10:  # Warning Sign (diamond)
            pts = np.array([[cx, cy - radius], [cx + radius, cy], [cx, cy + radius], [cx - radius, cy]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (240, 200, 20))
            cv2.polylines(img, [pts], True, (20, 20, 20), 2)

    def generate_video_sequence(self, num_frames: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates consecutive video frames with moving hazards to evaluate temporal smoothing and flicker.
        """
        sequence = []
        # Simulate a pothole approaching on near lane (moving down rows)
        start_row = 10
        col = 3
        for t in range(num_frames):
            curr_row = max(0, start_row - int(t / 2.5))
            coi_idx = curr_row * 8 + col
            # 80% chance detection, with 20% random dropout flicker to simulate edge camera noise
            flicker_active = random.random() > 0.15
            forced = [(coi_idx, 0)] if flicker_active else []
            img, target, _ = self.generate_frame(forced_hazards=forced)
            sequence.append((img, target))
        return sequence
