"""
YOLIC Paper-Style Interactive Visualizer (Streamlit)
Replicates the 104-CoI grid overlays from Su et al. (Image and Vision Computing 2024, Fig. 1 & Fig. 8):
- 8 top traffic sign cells + 96 perspective lower road cells.
- Semi-transparent green outlines for clear road cells.
- Transparent colored fill + border for active hazard cells.
- Class name and confidence score badges inside active cells.
- Single-core latency, FPS, and closed-loop actuator signals.
"""
from typing import Any, Dict, List, Optional, Tuple
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
    page_title="Extended YOLIC Edge Vision",
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
    1: (0, 165, 255),   # Speed Bump: Orange
    2: (128, 128, 128), # Manhole Cover: Slate
    3: (0, 215, 255),   # Pedestrian: Amber Gold
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


def draw_yolic_grid(
    image_bgr: np.ndarray,
    probabilities: np.ndarray,
    num_cois: int = 104,
    threshold: float = 0.5,
    show_road_grid: bool = True
) -> Tuple[np.ndarray, List[dict]]:
    """
    Renders paper-matching perspective cell grid and hazard overlays.
    - Green outlines for clear road cells (P_bg >= threshold).
    - Colored fills and badges for active hazard cells.
    """
    H, W, _ = image_bgr.shape
    polygons = get_coi_polygons(num_cois)

    overlay = image_bgr.copy()
    active_hazards = []

    # Probability slices
    # Shape: (N, 12)
    bg_probs = probabilities[:, 11]
    hazard_probs = probabilities[:, :11]

    for idx, poly_norm in enumerate(polygons):
        poly_px = (poly_norm * [W, H]).astype(np.int32)
        cx = int(np.mean(poly_px[:, 0]))
        cy = int(np.mean(poly_px[:, 1]))

        p_bg = bg_probs[idx]
        best_cls = int(np.argmax(hazard_probs[idx]))
        p_hazard = float(hazard_probs[idx, best_cls])

        # Active hazard condition (Paper definition: P_bg < 0.5 and P_hazard >= 0.5)
        is_hazard = (p_bg < threshold) and (p_hazard >= threshold)

        if is_hazard:
            # Active hazard cell
            cls_name = CLASS_NAMES[best_cls]
            color_bgr = CLASS_COLORS.get(best_cls, (0, 0, 255))

            # Semi-transparent colored fill
            cv2.fillPoly(overlay, [poly_px], color_bgr)
            # Solid bright border
            cv2.polylines(overlay, [poly_px], isClosed=True, color=(255, 255, 255), thickness=2)

            # Class label & confidence text
            label = f"{cls_name} {p_hazard:.2f}"
            font_scale = 0.35 if H <= 300 else 0.45
            cv2.putText(
                overlay,
                label,
                (max(5, cx - 35), max(15, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            active_hazards.append({
                "coi_id": idx,
                "class": cls_name,
                "conf": round(p_hazard, 3),
                "is_near_field": (idx <= 20)
            })

        elif show_road_grid:
            # Clear road cell: subtle semi-transparent green outline
            cv2.polylines(overlay, [poly_px], isClosed=True, color=(0, 200, 0), thickness=1)

    # Alpha blend overlay with base image
    blended = cv2.addWeighted(overlay, 0.45, image_bgr, 0.55, 0)

    # Re-draw the crisp boundaries of active hazard cells over blended image
    for h in active_hazards:
        poly_px = (polygons[h["coi_id"]] * [W, H]).astype(np.int32)
        color_bgr = CLASS_COLORS.get(CLASS_NAMES.index(h["class"]), (0, 0, 255))
        cv2.polylines(blended, [poly_px], isClosed=True, color=color_bgr, thickness=2)

    return blended, active_hazards


def main():
    st.title("🛴 Extended YOLIC: Edge Vision & Actuation System")
    st.caption("Implementation based on Su et al. (Image and Vision Computing 2024; doi:10.1016/j.imavis.2024.105095)")

    # Sidebar: Architectural Configurations
    st.sidebar.header("🔬 Model Configuration")
    num_cois = st.sidebar.selectbox("CoI Topology", [104, 256], index=0, help="104: Outdoor E-Scooter (8 sign + 96 road). 256: Cityscapes (16 sign + 240 road).")
    backbone_type = st.sidebar.selectbox("Backbone", ["shufflenet_v2", "mobilenet_v2"], index=0)

    st.sidebar.subheader("Novel Extensions")
    use_context = st.sidebar.checkbox("Inter-Cell Context Module (MHSA)", value=True, help="Multi-Head Self-Attention across CoIs")
    use_grouped = st.sidebar.checkbox("Grouped / Parameter-Bounded Head", value=True, help="Shared projection head invariant to N")

    st.sidebar.subheader("Temporal Consistency")
    use_smoother = st.sidebar.checkbox("Enable Temporal Smoother (EMA)", value=True)
    alpha = st.sidebar.slider("Smoothing Alpha (α)", min_value=0.5, max_value=0.99, value=0.8, step=0.05, help="S_t = α * P_t + (1 - α) * S_{t-1}")

    st.sidebar.subheader("Detection & Visualization")
    threshold = st.sidebar.slider("Confidence Threshold (τ)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    show_road_grid = st.sidebar.checkbox("Show Clear Road CoI Grid", value=True)

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

    # Image Preprocessing Transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Mode Selector
    mode = st.radio("Select Input Stream:", ["Synthetic Scene Generator", "Image Upload", "Live Stream / Sequence Simulation"], horizontal=True)

    generator = SyntheticSceneGenerator(num_cois=num_cois, num_classes=11)

    if mode == "Synthetic Scene Generator":
        st.markdown("#### Procedural Synthetic Road Scene Generator")
        col_ctrl, col_act = st.columns([1, 2])

        with col_ctrl:
            num_hazards = st.slider("Inject Random Hazards:", min_value=0, max_value=6, value=2)
            regen_btn = st.button("🔄 Generate New Scene")

        # Generate Frame
        img_rgb, target, ground_truth = generator.generate_frame(num_hazards=num_hazards)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Preprocess & Run Inference
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

        # Render Overlays
        rendered_bgr, active_hazards = draw_yolic_grid(
            img_bgr,
            probs,
            num_cois=num_cois,
            threshold=threshold,
            show_road_grid=show_road_grid
        )
        rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)

        # Actuator Logic
        actuation = actuator.evaluate(active_hazards)

        # Layout: Visualization and Telemetry
        col_img, col_metrics = st.columns([3, 2])

        with col_img:
            # Scaled up display for clarity
            display_img = cv2.resize(rendered_rgb, (560, 560), interpolation=cv2.INTER_NEAREST)
            st.image(display_img, caption="YOLIC 104-CoI Perspective Cell Grid Overlay", use_container_width=True)

        with col_metrics:
            # Actuator Status Banner
            if actuation["emergency_stop"]:
                st.error(f"🛑 **EMERGENCY STOP ENGAGED**\n\n{actuation['status_message']}")
            elif actuation["speed_warning"]:
                st.warning(f"⚠️ **SPEED REDUCTION WARNING**\n\n{actuation['status_message']}")
            else:
                st.success("✅ **PATH CLEAR - ALL SAFE**\n\nTarget Cruising Speed: 25 km/h")

            # Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Latency", f"{latency_ms:.1f} ms")
            m_col2.metric("Throughput", f"{fps:.1f} FPS")
            m_col3.metric("Active CoIs", len(active_hazards))

            # Active Hazards Table
            st.markdown("##### Detected Cell Hazards")
            if active_hazards:
                st.dataframe(active_hazards, use_container_width=True)
            else:
                st.info("No active hazards detected in Cells of Interest.")

    elif mode == "Image Upload":
        uploaded_file = st.file_uploader("Upload Test Road Image", type=["jpg", "jpeg", "png"])
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

            rendered_bgr, active_hazards = draw_yolic_grid(
                img_bgr,
                probs,
                num_cois=num_cois,
                threshold=threshold,
                show_road_grid=show_road_grid
            )
            rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
            actuation = actuator.evaluate(active_hazards)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(cv2.resize(rendered_rgb, (560, 560)), caption="YOLIC Perspective Overlays", use_container_width=True)
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

    elif mode == "Live Stream / Sequence Simulation":
        st.markdown("#### Consecutive Frame Sequence (Temporal Consistency Benchmark)")
        num_frames = st.slider("Sequence Length (Frames)", min_value=10, max_value=50, value=25)
        run_stream = st.button("▶️ Run Video Sequence Simulation")

        if run_stream:
            st.session_state.smoother.reset()
            video_seq = generator.generate_video_sequence(num_frames=num_frames)

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

                rendered_bgr, active_hazards = draw_yolic_grid(
                    f_img_bgr,
                    probs,
                    num_cois=num_cois,
                    threshold=threshold,
                    show_road_grid=show_road_grid
                )
                rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
                actuation = actuator.evaluate(active_hazards)

                display_img = cv2.resize(rendered_rgb, (560, 560), interpolation=cv2.INTER_NEAREST)
                frame_slot.image(display_img, caption=f"Frame {idx+1}/{num_frames} | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS)")

                if actuation["emergency_stop"]:
                    actuator_slot.error(f"🛑 Frame {idx+1}: {actuation['status_message']}")
                elif actuation["speed_warning"]:
                    actuator_slot.warning(f"⚠️ Frame {idx+1}: {actuation['status_message']}")
                else:
                    actuator_slot.success(f"✅ Frame {idx+1}: Path Clear (Cruising)")

                time.sleep(0.04)  # ~25 FPS playback


if __name__ == "__main__":
    main()
