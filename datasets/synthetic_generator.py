"""
Synthetic Road Scene & CoI Grid Generator for YOLIC (Photorealistic HUD Edition)
Generates:
1. Exact CoI polygon geometry (top sign cells + perspective lower road cells).
2. Photorealistic synthetic road images (224x224) with asphalt texturing, perspective lane lines,
   horizon skylines, roadside poles, and detailed hazard visual models.
3. Pre-configured real-world driving scenarios (Emergency Stop, Warning, Intersection, Cruising).
4. Continuous video sequence generator with moving road textures and hazards for realistic 30 FPS streaming.
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
    - 96 to 103 (8 cells): Top traffic sign cells arranged across the upper band.
    """
    polygons = []
    num_rows = 12
    num_cols = 8
    y_bumper = 0.98
    y_horizon = 0.35

    x_bottom_left, x_bottom_right = 0.06, 0.94
    x_top_left, x_top_right = 0.34, 0.66

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

            poly = np.array([pt_bl, pt_br, pt_tr, pt_tl], dtype=np.float32)
            polygons.append(poly)

    y_sign_top = 0.05
    y_sign_bottom = 0.25
    x_sign_start = 0.05
    x_sign_end = 0.95
    sign_cols = 8

    for c in range(sign_cols):
        x0 = x_sign_start + (c / sign_cols) * (x_sign_end - x_sign_start)
        x1 = x_sign_start + ((c + 1) / sign_cols) * (x_sign_end - x_sign_start)
        poly = np.array([
            [x0 + 0.005, y_sign_bottom],
            [x1 - 0.005, y_sign_bottom],
            [x1 - 0.005, y_sign_top],
            [x0 + 0.005, y_sign_top]
        ], dtype=np.float32)
        polygons.append(poly)

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

    for r in range(2):
        y0 = 0.24 - r * 0.10
        y1 = 0.14 - r * 0.10
        for c in range(8):
            x0 = 0.05 + (c / 8) * 0.90
            x1 = 0.05 + ((c + 1) / 8) * 0.90
            polygons.append(np.array([[x0 + 0.003, y0], [x1 - 0.003, y0], [x1 - 0.003, y1], [x0 + 0.003, y1]], dtype=np.float32))

    return polygons


def get_coi_polygons(num_cois: int = 104) -> List[np.ndarray]:
    if num_cois == 104:
        return generate_coi_polygons_104()
    elif num_cois == 256:
        return generate_coi_polygons_256()
    return generate_coi_polygons_104()


class SyntheticSceneGenerator:
    """
    Renders photorealistic synthetic road imagery and corresponding ground-truth CoI target matrices.
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
        self.total_classes = num_classes + 1
        self.bg_index = num_classes
        self.image_size = image_size
        self.polygons = get_coi_polygons(num_cois)

    def generate_frame(
        self,
        num_hazards: Optional[int] = None,
        forced_hazards: Optional[List[Tuple[int, int]]] = None,
        lane_offset: float = 0.0,
        time_sec: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Renders a photorealistic driving frame with realistic asphalt, horizon, road markings, and hazards.
        """
        H, W = self.image_size
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Sky & Horizon Gradient (Sunset / Daytime atmospheric perspective)
        y_horizon_px = int(0.35 * H)
        for y in range(y_horizon_px):
            ratio = y / y_horizon_px
            # Sky color gradient: deep atmospheric blue to warm haze near horizon
            r_val = int(140 + 75 * ratio)
            g_val = int(175 + 50 * ratio)
            b_val = int(225 - 20 * ratio)
            img[y, :] = (r_val, g_val, b_val)

        # Distant Skyline / Mountain Silhouette
        for x in range(W):
            mountain_h = int(10 * math.sin(x * 0.05) + 6 * math.cos(x * 0.12) + 8)
            img[max(0, y_horizon_px - mountain_h):y_horizon_px, x] = (110, 120, 140)

        # 2. Roadside Grass & Guardrails
        # Sidewalk & terrain with realistic lighting
        img[y_horizon_px:, :] = (45, 95, 45)  # Terrain grass

        # Asphalt Road Surface (Perspective trapezoid)
        road_pts = np.array([
            [int(0.06 * W), H],
            [int(0.94 * W), H],
            [int(0.66 * W), y_horizon_px],
            [int(0.34 * W), y_horizon_px]
        ], dtype=np.int32)
        
        # Asphalt base tone with vertical perspective shading (darker near camera)
        for y in range(y_horizon_px, H):
            t_road = (y - y_horizon_px) / (H - y_horizon_px)
            base_gray = int(60 + 15 * (1.0 - t_road))
            x_left = int((0.34 + t_road * (0.06 - 0.34)) * W)
            x_right = int((0.66 + t_road * (0.94 - 0.66)) * W)
            img[y, max(0, x_left):min(W, x_right)] = (base_gray, base_gray, base_gray + 3)

        # Asphalt grain texture noise
        noise = np.random.randint(-8, 8, (H, W, 3), dtype=np.int16)
        road_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(road_mask, [road_pts], 255)
        blended = np.clip(img.astype(np.int16) + (noise * (road_mask[:, :, None] > 0)), 0, 255).astype(np.uint8)
        img = blended

        # 3. Road Markings (Solid border lines + dashed center line with motion offset)
        # Left white shoulder line
        cv2.line(img, (int(0.08 * W), H), (int(0.35 * W), y_horizon_px), (230, 230, 235), 2, cv2.LINE_AA)
        # Right white shoulder line
        cv2.line(img, (int(0.92 * W), H), (int(0.65 * W), y_horizon_px), (230, 230, 235), 2, cv2.LINE_AA)

        # Dashed yellow lane divider with animation offset
        num_dashes = 10
        for step in range(num_dashes):
            norm_step = (step + (lane_offset % 1.0)) / num_dashes
            if norm_step >= 1.0:
                continue
            t0 = norm_step ** 1.3
            t1 = min(1.0, (norm_step + 0.05) ** 1.3)
            y0 = int(H - t0 * (H - y_horizon_px))
            y1 = int(H - t1 * (H - y_horizon_px))
            dash_w = max(1, int(3 * (1 - t0)))
            cv2.line(img, (int(0.50 * W), y0), (int(0.50 * W), y1), (245, 215, 45), dash_w, cv2.LINE_AA)

        # Roadside Streetlight Poles
        pole_x_left, pole_x_right = int(0.04 * W), int(0.96 * W)
        cv2.line(img, (pole_x_left, H), (pole_x_left, int(0.20 * H)), (80, 80, 85), 2, cv2.LINE_AA)
        cv2.line(img, (pole_x_left, int(0.20 * H)), (pole_x_left + 15, int(0.18 * H)), (90, 90, 95), 2, cv2.LINE_AA)
        cv2.circle(img, (pole_x_left + 15, int(0.18 * H)), 3, (255, 250, 200), -1)

        # 4. Target Matrix Initialization: all cells clear road initially
        target = np.zeros((self.num_cois, self.total_classes), dtype=np.float32)
        target[:, self.bg_index] = 1.0

        # 5. Inject Hazards
        hazard_list = []
        if forced_hazards is not None:
            chosen_hazards = forced_hazards
        else:
            k = random.randint(1, 3) if num_hazards is None else num_hazards
            chosen_cois = random.sample(range(self.num_cois), min(k, self.num_cois))
            chosen_hazards = []
            for coi_idx in chosen_cois:
                if coi_idx >= 96:
                    cls_id = random.choice([8, 9, 10])
                else:
                    cls_id = random.choice(list(range(8)))
                chosen_hazards.append((coi_idx, cls_id))

        for coi_idx, cls_id in chosen_hazards:
            poly_norm = self.polygons[coi_idx]
            poly_px = (poly_norm * [W, H]).astype(np.int32)
            cx = int(np.mean(poly_px[:, 0]))
            cy = int(np.mean(poly_px[:, 1]))
            bw = int(np.ptp(poly_px[:, 0]))
            bh = int(np.ptp(poly_px[:, 1]))

            target[coi_idx, cls_id] = 1.0
            target[coi_idx, self.bg_index] = 0.0

            cls_name = self.CLASS_NAMES[cls_id]
            self._render_realistic_hazard(img, cls_id, cx, cy, bw, bh)
            hazard_list.append({
                "coi_id": coi_idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy)
            })

        return img, target, hazard_list

    def _render_realistic_hazard(self, img: np.ndarray, cls_id: int, cx: int, cy: int, bw: int, bh: int):
        """Draws photorealistic visual markers for hazards with 3D shading, highlights, and shadow."""
        radius = max(3, min(bw, bh) // 3)

        if cls_id == 0:  # Pothole: Deep crater with fracture lines & inner shadow
            cv2.ellipse(img, (cx, cy), (radius, max(2, radius // 2)), 0, 0, 360, (20, 20, 22), -1)
            cv2.ellipse(img, (cx, cy - 1), (radius - 1, max(1, radius // 2 - 1)), 0, 0, 360, (10, 10, 12), -1)
            cv2.ellipse(img, (cx, cy), (radius, max(2, radius // 2)), 0, 0, 360, (40, 40, 42), 1)
            # Fracture cracks
            cv2.line(img, (cx - radius, cy), (cx - radius - 3, cy - 2), (25, 25, 25), 1)
            cv2.line(img, (cx + radius, cy), (cx + radius + 4, cy + 1), (25, 25, 25), 1)

        elif cls_id == 1:  # Speed Bump: 3D perspective yellow/black hazard striping
            bw_half = max(6, bw // 2 - 2)
            bh_half = max(3, bh // 3)
            pts_bump = np.array([
                [cx - bw_half, cy + bh_half], [cx + bw_half, cy + bh_half],
                [cx + int(bw_half * 0.8), cy - bh_half], [cx - int(bw_half * 0.8), cy - bh_half]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts_bump], (30, 215, 250)) # Bright reflective yellow
            # Black diagonal caution stripes
            for s in range(-bw_half, bw_half, 6):
                cv2.line(img, (cx + s, cy + bh_half), (cx + s + 3, cy - bh_half), (20, 20, 25), 2)
            cv2.polylines(img, [pts_bump], True, (20, 20, 20), 1)

        elif cls_id == 2:  # Manhole Cover: Metallic steel rim and cross-hatch grip
            cv2.circle(img, (cx, cy), radius, (85, 90, 95), -1)
            cv2.circle(img, (cx, cy), radius, (40, 40, 45), 1)
            cv2.circle(img, (cx, cy), max(1, radius - 2), (110, 115, 120), 1)
            cv2.line(img, (cx - radius // 2, cy), (cx + radius // 2, cy), (50, 50, 50), 1)
            cv2.line(img, (cx, cy - radius // 2), (cx, cy + radius // 2), (50, 50, 50), 1)

        elif cls_id == 3:  # Pedestrian: Walking human silhouette with head and torso
            head_r = max(2, radius // 3)
            cv2.circle(img, (cx, cy - radius), head_r, (230, 195, 160), -1) # Head
            # Torso (Jacket)
            cv2.rectangle(img, (cx - head_r, cy - radius + head_r), (cx + head_r, cy), (30, 80, 210), -1)
            # Legs
            cv2.line(img, (cx - 2, cy), (cx - 4, cy + radius // 2), (30, 30, 40), 2)
            cv2.line(img, (cx + 2, cy), (cx + 4, cy + radius // 2), (30, 30, 40), 2)

        elif cls_id == 4:  # Bicycle: Two spoked wheels + frame
            w_r = max(2, radius // 2)
            cv2.circle(img, (cx - w_r - 2, cy), w_r, (40, 40, 40), 1)
            cv2.circle(img, (cx + w_r + 2, cy), w_r, (40, 40, 40), 1)
            cv2.line(img, (cx - w_r - 2, cy), (cx, cy - w_r), (210, 40, 40), 2)
            cv2.line(img, (cx + w_r + 2, cy), (cx, cy - w_r), (210, 40, 40), 2)

        elif cls_id == 5:  # Car: 3D vehicle rear with glowing red taillights and window
            car_w = max(8, bw // 2)
            car_h = max(5, bh // 2)
            # Car body
            cv2.rectangle(img, (cx - car_w, cy - car_h), (cx + car_w, cy + car_h), (180, 30, 30), -1)
            # Rear windshield
            cv2.rectangle(img, (cx - car_w + 2, cy - car_h + 1), (cx + car_w - 2, cy), (40, 60, 80), -1)
            # Left & Right glowing taillights
            cv2.circle(img, (cx - car_w + 3, cy + car_h - 2), 2, (0, 0, 255), -1)
            cv2.circle(img, (cx + car_w - 3, cy + car_h - 2), 2, (0, 0, 255), -1)

        elif cls_id == 6:  # Debris: Scattered rocks / road obstacle
            for dx, dy in [(-radius // 2, 0), (radius // 3, -1), (0, radius // 3), (-radius // 4, radius // 4)]:
                cv2.circle(img, (cx + dx, cy + dy), max(1, radius // 3), (90, 80, 70), -1)

        elif cls_id == 7:  # Curb: Raised concrete edge barrier
            cv2.line(img, (cx - bw // 2, cy), (cx + bw // 2, cy), (210, 210, 215), 3)
            cv2.line(img, (cx - bw // 2, cy + 1), (cx + bw // 2, cy + 1), (110, 110, 115), 1)

        elif cls_id == 8:  # Stop Sign (Octagonal red with metal pole)
            cv2.line(img, (cx, cy + radius), (cx, cy + radius + 10), (120, 120, 125), 2)
            pts = np.array([
                [cx - radius, cy - radius // 2], [cx - radius // 2, cy - radius],
                [cx + radius // 2, cy - radius], [cx + radius, cy - radius // 2],
                [cx + radius, cy + radius // 2], [cx + radius // 2, cy + radius],
                [cx - radius // 2, cy + radius], [cx - radius, cy + radius // 2]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (225, 25, 25))
            cv2.polylines(img, [pts], True, (255, 255, 255), 1)

        elif cls_id == 9:  # Yield Sign (Triangle with red border)
            cv2.line(img, (cx, cy + radius), (cx, cy + radius + 8), (120, 120, 125), 2)
            pts = np.array([[cx - radius, cy - radius], [cx + radius, cy - radius], [cx, cy + radius]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (250, 250, 250))
            cv2.polylines(img, [pts], True, (220, 30, 30), 2)

        elif cls_id == 10:  # Warning Sign (Yellow diamond)
            cv2.line(img, (cx, cy + radius), (cx, cy + radius + 8), (120, 120, 125), 2)
            pts = np.array([[cx, cy - radius], [cx + radius, cy], [cx, cy + radius], [cx - radius, cy]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (250, 210, 20))
            cv2.polylines(img, [pts], True, (20, 20, 20), 2)

    def get_preset_scenario(self, scenario_name: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Returns authentic driving scenarios for repeatable live demonstrations.
        """
        if scenario_name == "Emergency Stop: Pothole & Pedestrian (< 2m)":
            # CoI 7 (row 0 near bumper) = Pothole (0), CoI 12 (row 1 near bumper) = Pedestrian (3)
            forced = [(7, 0), (12, 3)]
        elif scenario_name == "Speed Warning: Speed Bump & Ahead Car":
            # CoI 35 (row 4 mid-road) = Speed Bump (1), CoI 60 (row 7 far-road) = Car (5)
            forced = [(35, 1), (60, 5)]
        elif scenario_name == "Intersection: Stop Sign & Crossing Bicycle":
            # CoI 98 (top sign band) = Stop Sign (8), CoI 28 (row 3 mid-road) = Bicycle (4)
            forced = [(98, 8), (28, 4)]
        elif scenario_name == "Road Debris & Raised Curb Ahead":
            # CoI 22 (row 2 mid-road) = Debris (6), CoI 16 (row 2 right curb) = Curb (7)
            forced = [(22, 6), (16, 7)]
        else: # "Clear Path: Normal Cruising"
            forced = []

        return self.generate_frame(forced_hazards=forced)

    def generate_video_sequence(self, num_frames: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates consecutive video frames with moving road lane dashes and approaching hazards
        to evaluate temporal consistency and real-time playback.
        """
        sequence = []
        start_row = 10
        col = 3
        for t in range(num_frames):
            curr_row = max(0, start_row - int(t / 2.5))
            coi_idx = curr_row * 8 + col
            forced = [(coi_idx, 0)] if t > 3 else []
            # Animate road lane dash offset
            lane_offset = (t * 0.15) % 1.0
            img, target, _ = self.generate_frame(forced_hazards=forced, lane_offset=lane_offset)
            sequence.append((img, target))
        return sequence
