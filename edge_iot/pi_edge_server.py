"""
Ultra-Lightweight Raspberry Pi Edge Server for YOLIC
Runs inference locally on Raspberry Pi CPU and streams live annotated HUD video + telemetry
to any laptop browser over Wi-Fi / LAN with zero client setup.
"""

import os
import sys
import time
import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
from torchvision import transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.unified_yolic import UnifiedYOLIC
from models.temporal_smoother import TemporalSmoother
from datasets.synthetic_generator import get_coi_polygons
from edge_iot.actuator_logic import ActuatorLogic

# 11 Paper Classes
CLASS_NAMES = [
    "Bump", "Column", "Dent", "Fence", "Creature",
    "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone", "TrafficSign"
]
CLASS_COLORS = {
    0: (10, 249, 72),    # Bump (Green)
    1: (151, 157, 255),  # Column (Sky Blue)
    2: (134, 219, 61),   # Dent / Pothole (Lime)
    3: (52, 147, 26),    # Fence (Dark Green)
    4: (29, 178, 255),   # Creature / Pedestrian (Cyan-Orange)
    5: (31, 112, 255),   # Vehicle (Vivid Orange)
    6: (49, 210, 207),   # Wall (Teal)
    7: (23, 204, 146),   # Weed / Debris (Sea Green)
    8: (56, 56, 255),    # ZebraCrossing (Bright Red)
    9: (187, 212, 0),    # TrafficCone (Yellow-Cyan)
    10: (168, 153, 44),  # TrafficSign (Slate Blue)
    11: (0, 255, 0)      # Background / Clear (Green)
}

# Global frame buffer for streaming
latest_jpeg = None
latest_telemetry = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "status": "INITIALIZING",
    "target_speed": 25,
    "active_hazards": []
}
frame_lock = Lock()


def get_local_ip() -> str:
    """Detects local network IP of the Raspberry Pi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def render_yolic_hud(
    image_bgr: np.ndarray,
    probs: np.ndarray,
    polygons: List[np.ndarray],
    threshold: float = 0.50,
    fps: float = 30.0,
    latency_ms: float = 32.0
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Renders paper-matching 104-CoI perspective grid HUD with closed-loop actuation."""
    H, W = 480, 640
    hires = cv2.resize(image_bgr, (W, H))
    overlay = hires.copy()

    bg_probs = probs[:, 11]
    haz_probs = probs[:, :11]

    active_hazards = []

    # Corridor
    pts_corridor = np.array([
        [int(0.24 * W), H], [int(0.76 * W), H],
        [int(0.55 * W), int(0.35 * H)], [int(0.45 * W), int(0.35 * H)]
    ], dtype=np.int32)
    cv2.polylines(overlay, [pts_corridor], False, (255, 230, 80), 2, cv2.LINE_AA)

    for idx, poly in enumerate(polygons):
        poly_px = (poly * [W, H]).astype(np.int32)
        cx = int(np.mean(poly_px[:, 0]))
        cy = int(np.mean(poly_px[:, 1]))
        top_y = int(np.min(poly_px[:, 1]))

        p_bg = bg_probs[idx]
        best_cls = int(np.argmax(haz_probs[idx]))
        p_haz = float(haz_probs[idx, best_cls])

        # Anti-hallucination verification
        is_hazard = (p_bg < 0.50) and (p_haz >= threshold) and (p_haz > p_bg)

        if is_hazard:
            cls_name = CLASS_NAMES[best_cls]
            color = CLASS_COLORS.get(best_cls, (0, 0, 255))
            is_near = (cy > 0.60 * H) or (idx <= 20)

            cv2.fillPoly(overlay, [poly_px], color)
            active_hazards.append({
                "coi_id": idx,
                "class": cls_name,
                "conf": round(p_haz, 2),
                "is_near": is_near,
                "cx": cx,
                "top_y": top_y
            })
        else:
            cv2.polylines(overlay, [poly_px], True, (130, 130, 130), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.45, hires, 0.55, 0)

    # Crisp tags
    for h in active_hazards:
        poly_px = (polygons[h["coi_id"]] * [W, H]).astype(np.int32)
        color = CLASS_COLORS.get(CLASS_NAMES.index(h["class"]), (0, 0, 255))
        cv2.polylines(blended, [poly_px], True, color, 2, cv2.LINE_AA)
        tag = f"{h['class']} {int(h['conf']*100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bx = np.clip(h["cx"] - tw//2, 6, W - tw - 6)
        by = np.clip(h["top_y"] - 4, 25, H - 35)
        cv2.rectangle(blended, (bx - 2, by - th - 2), (bx + tw + 2, by + 2), (15, 15, 15), -1)
        cv2.putText(blended, tag, (bx, by - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Top Status Bar
    cv2.rectangle(blended, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(blended, f"PI-EDGE TINYML | 104-COI | INFER: {latency_ms:.1f}ms ({fps:.1f} FPS)", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # Bottom Actuator Bar
    cv2.rectangle(blended, (0, H - 30), (W, H), (15, 15, 15), -1)
    has_near = any(h["is_near"] for h in active_hazards)
    has_mid = len(active_hazards) > 0 and not has_near

    if has_near:
        status_msg = "ACTUATOR: [EMERGENCY BRAKE] Target speed: 0 km/h"
        status_color = (0, 0, 255)
        target_spd = 0
    elif has_mid:
        status_msg = "ACTUATOR: [SPEED WARNING] Target speed: 10 km/h"
        status_color = (0, 165, 255)
        target_spd = 10
    else:
        status_msg = "ACTUATOR: [ALL CLEAR] Cruising speed: 25 km/h"
        status_color = (0, 255, 0)
        target_spd = 25

    cv2.putText(blended, status_msg, (16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    actuation = {
        "emergency_stop": has_near,
        "speed_warning": has_mid,
        "target_speed": target_spd,
        "status_message": status_msg
    }

    return blended, active_hazards, actuation


def edge_inference_worker(camera_index: int = 0):
    """Background worker continuously reading camera, running inference, and encoding MJPEG."""
    global latest_jpeg, latest_telemetry

    print(f"Starting camera capture on device index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Warning: Camera index {camera_index} could not be opened. Using test stream generator.")

    # Load Model
    torch.set_num_threads(2)
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
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "outdoor_104coi", "best_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded calibrated Edge TinyML checkpoint from {ckpt_path}")
    model.eval()

    smoother = TemporalSmoother(alpha=0.8)
    polygons = get_coi_polygons(104)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sample_frame_idx = 0
    test_files = []
    real_dir = os.path.join(PROJECT_ROOT, "data", "real_author_dataset", "RGB")
    if os.path.exists(real_dir):
        test_files = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) if f.endswith(".png")]

    while True:
        t0 = time.perf_counter()
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
        else:
            # Fallback to cycling real author test frames if camera hardware is not plugged in
            if test_files:
                frame = cv2.imread(test_files[sample_frame_idx % len(test_files)])
                sample_frame_idx += 1
                time.sleep(0.04)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Preprocess and Infer
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(T.functional.to_pil_image(rgb)).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            raw_out = logits.squeeze(0).numpy()
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw_out, -20, 20)))

        probs = smoother.update(probs)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        rendered, active_hazards, actuation = render_yolic_hud(
            frame,
            probs,
            polygons,
            threshold=0.50,
            fps=fps,
            latency_ms=latency_ms
        )

        _, jpeg_bytes = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        with frame_lock:
            latest_jpeg = jpeg_bytes.tobytes()
            latest_telemetry = {
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "status": actuation["status_message"],
                "target_speed": actuation["target_speed"],
                "hazards_count": len(active_hazards),
                "active_hazards": active_hazards
            }


class StreamingHandler(BaseHTTPRequestHandler):
    """Serves the Web Dashboard and low-latency MJPEG video stream."""

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi Edge YOLIC HUD</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            background: #0e1117;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #262730;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
        .stream-card {{
            background: #1a1c23;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2d3139;
        }}
        .stream-img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
        }}
        .card {{
            background: #1a1c23;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #2d3139;
        }}
        .metric-row {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .metric {{
            background: #22252e;
            padding: 12px;
            border-radius: 6px;
            flex: 1;
            text-align: center;
        }}
        .metric-val {{
            font-size: 24px;
            font-weight: bold;
            color: #00ffaa;
        }}
        .metric-label {{
            font-size: 12px;
            color: #8b949e;
            margin-top: 4px;
        }}
        .actuator-box {{
            padding: 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .brake {{ background: #5a1010; color: #ff6b6b; border: 1px solid #ff4444; }}
        .warning {{ background: #5a3c10; color: #ffd166; border: 1px solid #ffaa00; }}
        .clear {{ background: #104a20; color: #06d6a0; border: 1px solid #00cc66; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 8px;
            border-bottom: 1px solid #2d3139;
            text-align: left;
        }}
        th {{ color: #8b949e; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h2>🛴 Raspberry Pi Edge YOLIC Vision HUD</h2>
            <div style="color: #8b949e; font-size: 14px;">Extreme Edge TinyML Object Localization & Classification on Edge Devices (Paper: Su et al. 2024)</div>
        </div>
        <div style="text-align: right; font-size: 13px; color: #00ffaa;">
            🟢 LIVE EDGE NODE CONNECTED
        </div>
    </div>

    <div class="grid">
        <div class="stream-card">
            <img class="stream-img" src="/stream.mjpg" alt="Live YOLIC Stream" />
        </div>
        <div>
            <div id="actuator-banner" class="actuator-box clear">ACTUATOR: INITIALIZING...</div>

            <div class="card">
                <div class="metric-row">
                    <div class="metric">
                        <div id="metric-fps" class="metric-val">--</div>
                        <div class="metric-label">INFERENCE FPS</div>
                    </div>
                    <div class="metric">
                        <div id="metric-lat" class="metric-val">-- ms</div>
                        <div class="metric-label">LATENCY</div>
                    </div>
                    <div class="metric">
                        <div id="metric-speed" class="metric-val">-- km/h</div>
                        <div class="metric-label">TARGET SPEED</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h4 style="margin-top: 0;">Active CoI Hazard Detections</h4>
                <table>
                    <thead>
                        <tr><th>CoI #</th><th>Class</th><th>Confidence</th><th>Field</th></tr>
                    </thead>
                    <tbody id="hazard-tbody">
                        <tr><td colspan="4" style="color: #8b949e;">No active hazards detected.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function updateTelemetry() {{
            try {{
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                
                document.getElementById('metric-fps').innerText = data.fps;
                document.getElementById('metric-lat').innerText = data.latency_ms + ' ms';
                document.getElementById('metric-speed').innerText = data.target_speed + ' km/h';

                const banner = document.getElementById('actuator-banner');
                banner.innerText = data.status;
                if (data.target_speed === 0) {{
                    banner.className = 'actuator-box brake';
                }} else if (data.target_speed === 10) {{
                    banner.className = 'actuator-box warning';
                }} else {{
                    banner.className = 'actuator-box clear';
                }}

                const tbody = document.getElementById('hazard-tbody');
                if (data.active_hazards && data.active_hazards.length > 0) {{
                    tbody.innerHTML = data.active_hazards.map(h => `
                        <tr>
                            <td>#${{h.coi_id + 1}}</td>
                            <td><strong>${{h.class}}</strong></td>
                            <td>${{Math.round(h.conf * 100)}}%</td>
                            <td>${{h.is_near ? '<span style="color: #ff6b6b;">Near (Brake)</span>' : '<span style="color: #ffd166;">Mid-Range</span>'}}</td>
                        </tr>
                    `).join('');
                }} else {{
                    tbody.innerHTML = '<tr><td colspan="4" style="color: #06d6a0;">✅ All clear. No obstacles in path.</td></tr>';
                }}
            }} catch (e) {{
                console.error(e);
            }}
        }}
        setInterval(updateTelemetry, 150);
    </script>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if latest_jpeg is None:
                            time.sleep(0.01)
                            continue
                        frame_data = latest_jpeg

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_data)))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)
            except Exception:
                pass

        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with frame_lock:
                telemetry_json = json.dumps(latest_telemetry)
            self.wfile.write(telemetry_json.encode("utf-8"))

        else:
            self.send_error(404)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge YOLIC Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP Server Port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    # Start inference in background thread
    worker_thread = Thread(target=edge_inference_worker, args=(args.camera,), daemon=True)
    worker_thread.start()

    local_ip = get_local_ip()
    server_address = ("0.0.0.0", args.port)
    httpd = HTTPServer(server_address, StreamingHandler)

    print("\n" + "=" * 65)
    print("🚀 RASPBERRY PI YOLIC EDGE SERVER RUNNING")
    print("=" * 65)
    print(f"📡 Local Pi Address:   http://localhost:{args.port}")
    print(f"💻 Open on your Laptop: http://{local_ip}:{args.port}")
    print(f"📹 Video Stream URL:   http://{local_ip}:{args.port}/stream.mjpg")
    print(f"📊 Telemetry API:      http://{local_ip}:{args.port}/api/telemetry")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Edge Server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
