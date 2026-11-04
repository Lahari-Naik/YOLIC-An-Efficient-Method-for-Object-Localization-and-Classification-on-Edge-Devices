"""
YOLIC Paper-Style Interactive Visualizer (Streamlit - High-Definition Autonomous Vehicle HUD)
Replicates the 104-CoI grid overlays from Su et al. (Image and Vision Computing 2024, Fig. 1 & Fig. 8)
with razor-sharp 640x640 Autonomous Driving Heads-Up Display (HUD) overlays:
- Crisp `● REC` live timecode counter & Edge AI Telemetry.
- Dynamic vehicle trajectory lines and perspective corridor.
- Distance-aware hazard badges (<1.8m Danger vs. >5m Warning).
- Real-time speedometer gauge with emergency brake strobe.
- Preset real-world road driving scenarios (Emergency Stop, Speed Warning, Intersection).
- Continuous real-time 30 FPS driving simulation.
"""
from typing import Any, Dict, List, Optional, Tuple
import datetime
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import streamlit as st

from models.unified_yolic import UnifiedYOLIC
from models.temporal_smoother import TemporalSmoother
from datasets.synthetic_generator import SyntheticSceneGenerator, get_coi_polygons
from edge_iot.actuator_logic import ActuatorLogic


# Page Configuration
st.set_page_config(
    page_title="Extended YOLIC: Edge Vision HUD",
    page_icon="🛴",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = [
    "Pothole", "Speed Bump", "Manhole Cover", "Pedestrian",
    "Bicycle", "Car", "Debris", "Curb", "Stop Sign",
    "Yield Sign", "Warning Sign"
]
BACKGROUND_NAME = "Clear Road"

CLASS_COLORS = {
    0: (0, 0, 230),     # Pothole: Crimson Red
    1: (0, 165, 255),   # Speed Bump: Amber Orange
    2: (128, 128, 128), # Manhole Cover: Slate
    3: (0, 215, 255),   # Pedestrian: High-vis Gold
    4: (238, 130, 238), # Bicycle: Violet
    5: (220, 20, 60),   # Car: Deep Red
    6: (19, 69, 139),   # Debris: Brown
    7: (211, 211, 211), # Curb: Light Grey
    8: (0, 0, 255),     # Stop Sign: Bright Red
    9: (50, 205, 50),   # Yield Sign: Lime Green
    10: (0, 215, 255),  # Warning Sign: Yellow-Gold
    11: (0, 255, 0)     # Clear Road: Green
}


@st.cache_resource
def load_yolic_model(
    num_cois: int,
    backbone_type: str,
    use_context_module: bool,
    use_grouped_head: bool
):
    """Loads and caches the Unified YOLIC model."""
    torch.set_num_threads(1)
    model = UnifiedYOLIC(
        num_cois=num_cois,
        num_classes=11,
        backbone_type=backbone_type,
        use_context_module=use_context_module,
        use_grouped_head=use_grouped_head,
        pretrained_backbone=False
    )
    model.eval()
    return model


def draw_yolic_hud_grid(
    image_bgr: np.ndarray,
    probabilities: np.ndarray,
    num_cois: int = 104,
    threshold: float = 0.5,
    show_road_grid: bool = True,
    fps: float = 31.4,
    latency_ms: float = 31.8,
    frame_idx: int = 1,
    render_dim: int = 640
) -> Tuple[np.ndarray, List[dict], dict]:
    """
    Renders paper-matching perspective cell grid and razor-sharp Autonomous Vehicle HUD telemetry.
    Upscales to render_dim (640x640) so graphics, borders, and text are crisp and unpixelated.
    """
    # 1. Upscale image to high resolution for sharp HUD rendering
    hires_img = cv2.resize(image_bgr, (render_dim, render_dim), interpolation=cv2.INTER_LINEAR)
    H, W = render_dim, render_dim
    polygons = get_coi_polygons(num_cois)

    overlay = hires_img.copy()
    active_hazards = []

    bg_probs = probabilities[:, 11]
    hazard_probs = probabilities[:, :11]

    # 2. Perspective Trajectory Guide (Vehicle Driving Corridor)
    pts_corridor = np.array([
        [int(0.25 * W), H],
        [int(0.75 * W), H],
        [int(0.55 * W), int(0.38 * H)],
        [int(0.45 * W), int(0.38 * H)]
    ], dtype=np.int32)
    cv2.polylines(overlay, [pts_corridor], isClosed=False, color=(255, 230, 80), thickness=2, lineType=cv2.LINE_AA)

    for idx, poly_norm in enumerate(polygons):
        poly_px = (poly_norm * [W, H]).astype(np.int32)
        cx = int(np.mean(poly_px[:, 0]))
        cy = int(np.mean(poly_px[:, 1]))

        p_bg = bg_probs[idx]
        best_cls = int(np.argmax(hazard_probs[idx]))
        p_hazard = float(hazard_probs[idx, best_cls])

        is_hazard = (p_bg < threshold) and (p_hazard >= threshold)

        if is_hazard:
            cls_name = CLASS_NAMES[best_cls]
            color_bgr = CLASS_COLORS.get(best_cls, (0, 0, 255))
            is_near_field = (idx <= 20)

            # Colored semi-transparent highlight fill
            cv2.fillPoly(overlay, [poly_px], color_bgr)
            # Solid bright border
            cv2.polylines(overlay, [poly_px], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            row_idx = idx // 8 if idx < 96 else 99
            if is_near_field:
                dist_str = "< 1.8m [DANGER]"
            else:
                dist_est = 2.0 + (row_idx * 0.9)
                dist_str = f"~{dist_est:.1f}m"

            # Crisp Badge Label inside or above cell
            label = f"{cls_name} {int(p_hazard*100)}%"
            # Badge background capsule
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(overlay, (cx - tw // 2 - 4, cy - th - 4), (cx + tw // 2 + 4, cy + 4), (20, 20, 25), -1)
            cv2.putText(
                overlay,
                label,
                (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            active_hazards.append({
                "coi_id": idx,
                "class": cls_name,
                "conf": round(p_hazard, 3),
                "is_near_field": is_near_field,
                "distance": dist_str
            })

        elif show_road_grid:
            # Clear road cell: subtle green perspective outline
            cv2.polylines(overlay, [poly_px], isClosed=True, color=(0, 225, 0), thickness=1, lineType=cv2.LINE_AA)

    # Alpha blend overlay with base image
    blended = cv2.addWeighted(overlay, 0.55, hires_img, 0.45, 0)

    # Crisp borders on active hazard cells
    for h in active_hazards:
        poly_px = (polygons[h["coi_id"]] * [W, H]).astype(np.int32)
        color_bgr = CLASS_COLORS.get(CLASS_NAMES.index(h["class"]), (0, 0, 255))
        cv2.polylines(blended, [poly_px], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    # 3. Autonomous Vehicle HUD Top & Bottom Bars
    # Top Bar (Dark semi-transparent header)
    top_bar_h = 32
    cv2.rectangle(blended, (0, 0), (W, top_bar_h), (18, 20, 24), -1)
    cv2.line(blended, (0, top_bar_h), (W, top_bar_h), (60, 220, 100), 2)

    # Left: REC Indicator & Timecode
    cv2.circle(blended, (18, 16), 6, (0, 0, 255), -1)
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    time_str = f"REC  {now_time}.{(frame_idx % 30):02d}  |  CAM: SCOOTER_FRONT_01"
    cv2.putText(blended, time_str, (32, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 245, 250), 1, cv2.LINE_AA)

    # Right: Edge CPU Telemetry
    cpu_info = f"EDGE AI: 1-CORE CPU | {latency_ms:.1f}ms ({fps:.1f} FPS)"
    (cw, ch), _ = cv2.getTextSize(cpu_info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(blended, cpu_info, (W - cw - 16, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 220), 1, cv2.LINE_AA)

    # Bottom Bar (Dark footer)
    bot_bar_h = 34
    cv2.rectangle(blended, (0, H - bot_bar_h), (W, H), (18, 20, 24), -1)
    cv2.line(blended, (0, H - bot_bar_h), (W, H - bot_bar_h), (60, 220, 100), 2)

    # Actuator Signal Status
    has_near_hazard = any(h["is_near_field"] for h in active_hazards)
    has_hazards = len(active_hazards) > 0

    if has_near_hazard:
        status_txt = "ACTUATOR: [EMERGENCY BRAKE ENGAGED] (0 km/h)"
        status_color = (40, 40, 255) # Red
    elif has_hazards:
        status_txt = "ACTUATOR: [SPEED WARNING REDUCTION] (10 km/h)"
        status_color = (0, 220, 255) # Yellow
    else:
        status_txt = "ACTUATOR: [ALL CLEAR CRUISING] (25 km/h)"
        status_color = (50, 255, 60) # Green

    cv2.putText(blended, status_txt, (18, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)

    coi_badge = f"{num_cois} CoIs | ACTIVE HAZARDS: {len(active_hazards)}"
    (bw, bh), _ = cv2.getTextSize(coi_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(blended, coi_badge, (W - bw - 16, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    actuator_decision = {
        "emergency_stop": has_near_hazard,
        "speed_warning": has_hazards and not has_near_hazard,
        "target_speed_kmh": 0.0 if has_near_hazard else (10.0 if has_hazards else 25.0)
    }

    return blended, active_hazards, actuator_decision


def main():
    st.title("🛴 Extended YOLIC: Edge Vision HUD & Actuation")
    st.caption("Live Autonomous Vehicle HUD Edition | Paper: Su et al. (Image and Vision Computing 2024; doi:10.1016/j.imavis.2024.105095)")

    # Sidebar: Architectural Configurations
    st.sidebar.header("🔬 Model Configuration")
    num_cois = st.sidebar.selectbox("CoI Topology", [104, 256], index=0, help="104: Outdoor E-Scooter. 256: Cityscapes.")
    backbone_type = st.sidebar.selectbox("Backbone", ["shufflenet_v2", "mobilenet_v2"], index=0)

    st.sidebar.subheader("Novel Extensions")
    use_context = st.sidebar.checkbox("Inter-Cell Context Module (MHSA)", value=True)
    use_grouped = st.sidebar.checkbox("Grouped / Parameter-Bounded Head", value=True)

    st.sidebar.subheader("Temporal Consistency")
    use_smoother = st.sidebar.checkbox("Enable Temporal Smoother (EMA)", value=True)
    alpha = st.sidebar.slider("Smoothing Alpha (α)", min_value=0.5, max_value=0.99, value=0.8, step=0.05)

    st.sidebar.subheader("Detection & HUD Overlays")
    threshold = st.sidebar.slider("Confidence Threshold (τ)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    show_road_grid = st.sidebar.checkbox("Show Perspective CoI Grid", value=True)

    # Load Model
    model = load_yolic_model(num_cois, backbone_type, use_context, use_grouped)
    actuator = ActuatorLogic(near_field_max_idx=20 if num_cois == 104 else 40)

    # Session State for Temporal Smoother
    if "smoother" not in st.session_state:
        st.session_state.smoother = TemporalSmoother(alpha=alpha)
        st.session_state.current_num_cois = num_cois
    else:
        st.session_state.smoother.alpha = alpha
        if st.session_state.get("current_num_cois") != num_cois:
            st.session_state.smoother.reset()
            st.session_state.current_num_cois = num_cois

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Mode Selector
    mode = st.radio(
        "Select Perception Stream:",
        ["Real-World Driving Scenarios (Presets)", "Live Drive Simulation (Continuous 30 FPS)", "Upload Custom Road Image"],
        horizontal=True
    )

    generator = SyntheticSceneGenerator(num_cois=num_cois, num_classes=11)

    if mode == "Real-World Driving Scenarios (Presets)":
        st.markdown("#### Autonomous Micro-Mobility Perception Scenarios")
        
        col_preset, col_ctrl = st.columns([2, 1])
        with col_preset:
            scenario_choice = st.selectbox(
                "Choose Driving Preset Scenario:",
                [
                    "Emergency Stop: Pothole & Pedestrian (< 2m)",
                    "Speed Warning: Speed Bump & Ahead Car",
                    "Intersection: Stop Sign & Crossing Bicycle",
                    "Road Debris & Raised Curb Ahead",
                    "Clear Path: Normal Cruising"
                ]
            )
        with col_ctrl:
            st.write("")
            st.write("")
            regen = st.button("🔄 Reload Scene & Re-Infer")

        # Get Preset Scenario Frame
        img_rgb, target, _ = generator.get_preset_scenario(scenario_choice)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Preprocess & Run Model Inference
        img_pil = Image.fromarray(img_rgb)
        tensor = transform(img_pil).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].numpy()

        if use_smoother:
            probs = st.session_state.smoother.update(probs)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        # Render Authentic HD HUD
        rendered_bgr, active_hazards, actuation_dec = draw_yolic_hud_grid(
            img_bgr,
            probs,
            num_cois=num_cois,
            threshold=threshold,
            show_road_grid=show_road_grid,
            fps=fps,
            latency_ms=latency_ms
        )
        rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

        actuation = actuator.evaluate(active_hazards)

        # Layout: Visualization and Telemetry
        col_img, col_metrics = st.columns([3, 2])

        with col_img:
            st.image(rendered_rgb, caption="Live YOLIC Edge Vision Camera HUD Overlay (640x640 HD)", use_container_width=True)

        with col_metrics:
            # Actuator Status Banner
            if actuation["emergency_stop"]:
                st.error(f"🛑 **EMERGENCY STOP ENGAGED**\n\n{actuation['status_message']}")
            elif actuation["speed_warning"]:
                st.warning(f"⚠️ **SPEED REDUCTION WARNING**\n\n{actuation['status_message']}")
            else:
                st.success("✅ **PATH CLEAR - ALL SAFE**\n\nTarget Cruising Speed: 25 km/h")

            # Metrics Card
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Latency", f"{latency_ms:.1f} ms")
            m_col2.metric("Throughput", f"{fps:.1f} FPS")
            m_col3.metric("Target Speed", f"{actuation['target_speed_kmh']} km/h")

            # Active Hazards Table
            st.markdown("##### Detected Obstacles in Path")
            if active_hazards:
                st.dataframe(active_hazards, use_container_width=True)
            else:
                st.info("No active hazards detected in Cells of Interest.")

    elif mode == "Live Drive Simulation (Continuous 30 FPS)":
        st.markdown("#### Continuous Real-Time Autonomous Drive Feed")
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            stream_len = st.slider("Drive Stream Duration (Frames):", min_value=15, max_value=60, value=35)
            start_sim = st.button("▶️ Start Live Drive Feed")

        if start_sim:
            st.session_state.smoother.reset()
            video_seq = generator.generate_video_sequence(num_frames=stream_len)

            frame_slot = st.empty()
            telemetry_slot = st.empty()
            actuator_slot = st.empty()

            for idx, (f_img_rgb, _) in enumerate(video_seq):
                f_img_bgr = cv2.cvtColor(f_img_rgb, cv2.COLOR_RGB2BGR)
                f_pil = Image.fromarray(f_img_rgb)
                tensor = transform(f_pil).unsqueeze(0)

                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)[0].numpy()

                if use_smoother:
                    probs = st.session_state.smoother.update(probs)
                t1 = time.perf_counter()

                latency_ms = (t1 - t0) * 1000.0
                fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

                rendered_bgr, active_hazards, actuation_dec = draw_yolic_hud_grid(
                    f_img_bgr,
                    probs,
                    num_cois=num_cois,
                    threshold=threshold,
                    show_road_grid=show_road_grid,
                    fps=fps,
                    latency_ms=latency_ms,
                    frame_idx=idx + 1
                )
                rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
                actuation = actuator.evaluate(active_hazards)

                frame_slot.image(rendered_rgb, caption=f"Live Camera Stream [Frame {idx+1}/{stream_len}] | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS)")

                if actuation["emergency_stop"]:
                    actuator_slot.error(f"🛑 Frame {idx+1}: {actuation['status_message']}")
                elif actuation["speed_warning"]:
                    actuator_slot.warning(f"⚠️ Frame {idx+1}: {actuation['status_message']}")
                else:
                    actuator_slot.success(f"✅ Frame {idx+1}: Path Clear (Cruising at 25 km/h)")

                time.sleep(0.033) # 30 FPS playback

    elif mode == "Upload Custom Road Image":
        uploaded_file = st.file_uploader("Upload Dashcam / Road Photo", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_raw = Image.open(uploaded_file).convert("RGB")
            img_resized = image_raw.resize((224, 224))
            img_np = np.array(img_resized)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            tensor = transform(img_resized).unsqueeze(0)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0].numpy()
            if use_smoother:
                probs = st.session_state.smoother.update(probs)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            rendered_bgr, active_hazards, actuation_dec = draw_yolic_hud_grid(
                img_bgr,
                probs,
                num_cois=num_cois,
                threshold=threshold,
                show_road_grid=show_road_grid,
                fps=fps,
                latency_ms=latency_ms
            )
            rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
            actuation = actuator.evaluate(active_hazards)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(rendered_rgb, caption="Live HD HUD Overlays", use_container_width=True)
            with c2:
                if actuation["emergency_stop"]:
                    st.error(f"🛑 {actuation['status_message']}")
                elif actuation["speed_warning"]:
                    st.warning(f"⚠️ {actuation['status_message']}")
                else:
                    st.success("✅ Path Clear - Cruising Speed: 25 km/h")

                st.metric("Latency", f"{latency_ms:.1f} ms")
                st.metric("FPS", f"{fps:.1f}")
                if active_hazards:
                    st.dataframe(active_hazards, use_container_width=True)


if __name__ == "__main__":
    main()
