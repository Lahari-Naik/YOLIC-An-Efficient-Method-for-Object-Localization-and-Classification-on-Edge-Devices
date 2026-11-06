"""
Synthetic Road Scene & CoI Grid Generator for YOLIC (Bold High-Contrast Edition)
Generates:
1. Exact CoI polygon geometry (top sign cells + perspective lower road cells).
2. Bold, high-contrast road hazards (clearly visible potholes, pedestrians, speed bumps, cars).
3. Hazards positioned in rows 1 to 9 so they are never obscured by HUD overlays or vehicle bumpers.
4. Pre-configured driving scenarios (Clear Path, Emergency Stop, Speed Warning).
"""
from typing import Dict, List, Optional, Tuple
import math
import random
import cv2
import numpy as np


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
    # Keep road cells slightly above the absolute bottom (0.94) to leave room for HUD status bars
    y_bumper = 0.93
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

    # 8 Top sign cells
    y_sign_top = 0.06
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
    polygons = []
    num_rows = 20
    num_cols = 12
    y_bumper = 0.93
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
        "Bump", "Column", "Dent", "Fence", "Creature",
        "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone", "TrafficSign"
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
        lane_offset: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Renders a driving frame with realistic asphalt, horizon, road markings, and bold hazards.
        """
        H, W = self.image_size
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Sky & Horizon Gradient
        y_horizon_px = int(0.35 * H)
        for y in range(y_horizon_px):
            ratio = y / y_horizon_px
            r_val = int(140 + 75 * ratio)
            g_val = int(175 + 50 * ratio)
            b_val = int(225 - 20 * ratio)
            img[y, :] = (r_val, g_val, b_val)

        # Distant Skyline / Mountain Silhouette
        for x in range(W):
            mountain_h = int(10 * math.sin(x * 0.05) + 6 * math.cos(x * 0.12) + 8)
            img[max(0, y_horizon_px - mountain_h):y_horizon_px, x] = (110, 120, 140)

        # 2. Roadside Grass
        img[y_horizon_px:, :] = (45, 95, 45)

        # Asphalt Road Surface
        road_pts = np.array([
            [int(0.06 * W), H],
            [int(0.94 * W), H],
            [int(0.66 * W), y_horizon_px],
            [int(0.34 * W), y_horizon_px]
        ], dtype=np.int32)

        # Asphalt base tone with vertical perspective shading
        for y in range(y_horizon_px, H):
            t_road = (y - y_horizon_px) / (H - y_horizon_px)
            base_gray = int(60 + 15 * (1.0 - t_road))
            x_left = int((0.34 + t_road * (0.06 - 0.34)) * W)
            x_right = int((0.66 + t_road * (0.94 - 0.66)) * W)
            img[y, max(0, x_left):min(W, x_right)] = (base_gray, base_gray, base_gray + 3)

        # Asphalt noise texture
        noise = np.random.randint(-6, 6, (H, W, 3), dtype=np.int16)
        road_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(road_mask, [road_pts], 255)
        blended = np.clip(img.astype(np.int16) + (noise * (road_mask[:, :, None] > 0)), 0, 255).astype(np.uint8)
        img = blended

        # 3. Road Markings
        cv2.line(img, (int(0.08 * W), H), (int(0.35 * W), y_horizon_px), (235, 235, 240), 2, cv2.LINE_AA)
        cv2.line(img, (int(0.92 * W), H), (int(0.65 * W), y_horizon_px), (235, 235, 240), 2, cv2.LINE_AA)

        # Dashed yellow center line
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

        # 4. Target Matrix: 100% clear road initially
        target = np.zeros((self.num_cois, self.total_classes), dtype=np.float32)
        target[:, self.bg_index] = 1.0

        # 5. Inject Hazards
        hazard_list = []
        if forced_hazards is not None:
            chosen_hazards = forced_hazards
        elif num_hazards and num_hazards > 0:
            chosen_cois = random.sample(range(16, min(80, self.num_cois)), num_hazards)
            chosen_hazards = [(idx, random.choice([0, 1, 3, 5])) for idx in chosen_cois]
        else:
            chosen_hazards = []

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
            self._render_bold_hazard(img, cls_id, cx, cy, bw, bh)
            hazard_list.append({
                "coi_id": coi_idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (cx, cy)
            })

        return img, target, hazard_list

    def _render_bold_hazard(self, img: np.ndarray, cls_id: int, cx: int, cy: int, bw: int, bh: int):
        """Draws bold, unmistakable, high-contrast visual hazard objects matching the 11 paper classes."""
        w_size = max(14, bw // 2)
        h_size = max(10, bh // 2)

        if cls_id == 0:  # Bump: Speed Bump (yellow & black diagonal hazard stripes)
            pts_bump = np.array([
                [cx - w_size - 4, cy + h_size // 2], [cx + w_size + 4, cy + h_size // 2],
                [cx + w_size, cy - h_size // 2], [cx - w_size, cy - h_size // 2]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts_bump], (20, 230, 255))
            for s in range(-w_size, w_size + 1, 6):
                cv2.line(img, (cx + s, cy + h_size // 2), (cx + s + 3, cy - h_size // 2), (15, 15, 20), 2)
            cv2.polylines(img, [pts_bump], True, (0, 0, 0), 1)

        elif cls_id == 1:  # Column: Vertical pole / lamppost / pillar
            col_w = max(4, w_size // 3)
            col_h = max(24, h_size * 2)
            cv2.rectangle(img, (cx - col_w, cy - col_h), (cx + col_w, cy + h_size // 2), (160, 165, 170), -1)
            cv2.rectangle(img, (cx - col_w, cy - col_h), (cx + col_w, cy + h_size // 2), (60, 60, 65), 1)
            cv2.circle(img, (cx, cy - col_h), col_w + 2, (220, 220, 230), -1)

        elif cls_id == 2:  # Dent: Pothole / road asphalt crater
            cv2.ellipse(img, (cx, cy), (w_size, max(4, h_size // 2)), 0, 0, 360, (15, 15, 18), -1)
            cv2.ellipse(img, (cx, cy), (w_size, max(4, h_size // 2)), 0, 0, 360, (180, 180, 190), 2)
            cv2.line(img, (cx - w_size, cy), (cx - w_size - 4, cy - 2), (200, 200, 210), 1)
            cv2.line(img, (cx + w_size, cy), (cx + w_size + 4, cy + 2), (200, 200, 210), 1)

        elif cls_id == 3:  # Fence: Railing / barrier pickets
            fence_w = max(18, w_size * 2)
            fence_h = max(12, h_size)
            # Horizontal rails
            cv2.line(img, (cx - fence_w // 2, cy - fence_h // 2), (cx + fence_w // 2, cy - fence_h // 2), (80, 140, 180), 2)
            cv2.line(img, (cx - fence_w // 2, cy + fence_h // 2), (cx + fence_w // 2, cy + fence_h // 2), (80, 140, 180), 2)
            # Vertical pickets
            for px in range(cx - fence_w // 2, cx + fence_w // 2 + 1, 6):
                cv2.line(img, (px, cy - fence_h // 2 - 3), (px, cy + fence_h // 2 + 4), (90, 150, 190), 2)

        elif cls_id == 4:  # Creature: Person / Pedestrian (Head, torso, legs / upper body)
            head_r = max(4, h_size // 3)
            # Head with hair/skin tone
            cv2.circle(img, (cx, cy - h_size - head_r), head_r, (215, 185, 160), -1)
            cv2.circle(img, (cx, cy - h_size - head_r), head_r, (60, 40, 30), 1)
            # Torso / Jacket (High-contrast clothing)
            cv2.rectangle(img, (cx - head_r - 2, cy - h_size), (cx + head_r + 2, cy), (0, 140, 255), -1)
            cv2.line(img, (cx - head_r - 2, cy - h_size // 2), (cx + head_r + 2, cy - h_size // 2), (255, 255, 255), 1)
            # Walking legs / lower body
            cv2.line(img, (cx - 2, cy), (cx - 4, cy + h_size), (40, 40, 50), 3)
            cv2.line(img, (cx + 2, cy), (cx + 5, cy + h_size), (40, 40, 50), 3)

        elif cls_id == 5:  # Vehicle: Car / van rear with windshield & taillights
            car_w = max(16, w_size)
            car_h = max(10, h_size)
            cv2.rectangle(img, (cx - car_w, cy - car_h), (cx + car_w, cy + car_h), (180, 30, 30), -1)
            cv2.rectangle(img, (cx - car_w + 3, cy - car_h + 1), (cx + car_w - 3, cy - 1), (30, 50, 70), -1)
            cv2.circle(img, (cx - car_w + 4, cy + car_h - 3), 3, (0, 0, 255), -1)
            cv2.circle(img, (cx + car_w - 4, cy + car_h - 3), 3, (0, 0, 255), -1)
            cv2.rectangle(img, (cx - 4, cy + car_h - 4), (cx + 4, cy + car_h - 1), (255, 255, 255), -1)

        elif cls_id == 6:  # Wall: Concrete barrier / brick wall segment
            wall_w = max(20, w_size * 2)
            wall_h = max(14, h_size)
            cv2.rectangle(img, (cx - wall_w // 2, cy - wall_h // 2), (cx + wall_w // 2, cy + wall_h // 2), (130, 120, 110), -1)
            cv2.rectangle(img, (cx - wall_w // 2, cy - wall_h // 2), (cx + wall_w // 2, cy + wall_h // 2), (60, 55, 50), 1)
            for brick_y in range(cy - wall_h // 2 + 4, cy + wall_h // 2, 4):
                cv2.line(img, (cx - wall_w // 2, brick_y), (cx + wall_w // 2, brick_y), (170, 160, 150), 1)

        elif cls_id == 7:  # Weed: Roadside vegetation / green shrub / leaves
            for angle in [0, 45, 90, 135, 180]:
                rad = math.radians(angle)
                lx = int(cx + w_size * 0.6 * math.cos(rad))
                ly = int(cy - h_size * 0.6 * math.sin(rad))
                cv2.ellipse(img, (lx, ly), (w_size // 3, h_size // 2), angle, 0, 360, (25, 140, 40), -1)
            cv2.circle(img, (cx, cy), max(3, w_size // 4), (15, 110, 30), -1)

        elif cls_id == 8:  # ZebraCrossing: White crosswalk stripes
            stripe_w = max(6, w_size // 2)
            stripe_h = max(12, h_size)
            for dx in [-stripe_w * 2, 0, stripe_w * 2]:
                pts = np.array([
                    [cx + dx - stripe_w // 2, cy - stripe_h // 2],
                    [cx + dx + stripe_w // 2, cy - stripe_h // 2],
                    [cx + dx + stripe_w // 2, cy + stripe_h // 2],
                    [cx + dx - stripe_w // 2, cy + stripe_h // 2]
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], (245, 245, 245))

        elif cls_id == 9:  # TrafficCone: Bright orange cone with white reflective band
            cone_w = max(8, w_size // 2)
            cone_h = max(16, h_size * 2)
            pts_cone = np.array([
                [cx - cone_w, cy + cone_h // 2],
                [cx + cone_w, cy + cone_h // 2],
                [cx, cy - cone_h // 2]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts_cone], (0, 120, 255))
            # White reflective band across middle
            cv2.line(img, (cx - cone_w // 2, cy), (cx + cone_w // 2, cy), (255, 255, 255), max(2, cone_h // 6))
            # Base plate
            cv2.rectangle(img, (cx - cone_w - 2, cy + cone_h // 2 - 2), (cx + cone_w + 2, cy + cone_h // 2 + 2), (0, 90, 210), -1)

        elif cls_id == 10:  # TrafficSign: Octagonal stop or warning sign on post
            cv2.line(img, (cx, cy + h_size), (cx, cy + h_size + 16), (140, 140, 145), 2)
            r_sign = max(7, w_size // 2)
            pts = np.array([
                [cx - r_sign, cy - r_sign // 2], [cx - r_sign // 2, cy - r_sign],
                [cx + r_sign // 2, cy - r_sign], [cx + r_sign, cy - r_sign // 2],
                [cx + r_sign, cy + r_sign // 2], [cx + r_sign // 2, cy + r_sign],
                [cx - r_sign // 2, cy + r_sign], [cx - r_sign, cy + r_sign // 2]
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (230, 20, 20))
            cv2.polylines(img, [pts], True, (255, 255, 255), 1)

    def get_preset_scenario(self, scenario_name: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Returns clean, calibrated driving scenarios where obstacles are clearly visible
        and positioned away from bottom status bar cutoffs.
        """
        if scenario_name == "Clear Path: Normal Cruising":
            # 100% clean road: 0 hazards!
            forced = []
        elif scenario_name == "Emergency Stop: Pedestrian Ahead (< 2m)":
            # CoI 11 (row 1, center-left, right in front of scooter): Creature / Pedestrian (4)
            forced = [(11, 4)]
        elif scenario_name == "Emergency Stop: Deep Pothole in Near-Lane":
            # CoI 12 (row 1, center-right): Dent / Pothole (2)
            forced = [(12, 2)]
        elif scenario_name == "Speed Warning: Speed Bump Ahead (~5m)":
            # CoI 35 (row 4, mid-lane): Bump / Speed Bump (0)
            forced = [(35, 0)]
        elif scenario_name == "Speed Warning: Car Ahead in Mid-Distance":
            # CoI 43 (row 5, center lane): Vehicle (5)
            forced = [(43, 5)]
        elif scenario_name == "Intersection: Overhead Stop Sign":
            # CoI 99 (top sign band): TrafficSign (10)
            forced = [(99, 10)]
        else:
            forced = []

        return self.generate_frame(forced_hazards=forced)

    def generate_video_sequence(self, num_frames: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
        sequence = []
        start_row = 9
        col = 3
        for t in range(num_frames):
            curr_row = max(1, start_row - int(t / 3.0))
            coi_idx = curr_row * 8 + col
            # Obstacle appears after frame 3
            forced = [(coi_idx, 0)] if t > 3 else []
            lane_offset = (t * 0.15) % 1.0
            img, target, _ = self.generate_frame(forced_hazards=forced, lane_offset=lane_offset)
            sequence.append((img, target))
        return sequence
