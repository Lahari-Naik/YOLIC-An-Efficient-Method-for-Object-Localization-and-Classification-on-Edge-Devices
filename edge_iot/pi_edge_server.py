"""
Ultra-Lightweight Raspberry Pi Edge Server for YOLIC
Runs inference locally on Raspberry Pi CPU.
Supports BOTH:
1. Laptop's Webcam (streams frames from laptop browser to Pi for edge inference)
2. Pi's Own Camera (CSI Camera Module or USB webcam on Pi)
"""

import os
import sys
import time
import json
import base64
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock
from typing import Optional, List, Dict, Tuple, Any

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

# Global Pipeline State
global_model = None
global_smoother = None
global_polygons = None
global_transform = None

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


def init_yolic_pipeline():
    """Initializes the YOLIC model and transforms once at startup."""
    global global_model, global_smoother, global_polygons, global_transform

    torch.set_num_threads(2)
    global_model = UnifiedYOLIC(
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
        global_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded calibrated Edge TinyML checkpoint from {ckpt_path}")
    global_model.eval()

    global_smoother = TemporalSmoother(alpha=0.8)
    global_polygons = get_coi_polygons(104)

    global_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def render_yolic_hud(
    image_bgr: np.ndarray,
    probs: np.ndarray,
    polygons: List[np.ndarray],
    threshold: float = 0.50,
    fps: float = 30.0,
    latency_ms: float = 32.0,
    device_label: str = "PI-EDGE TINYML"
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

        # Strict Anti-Hallucination Verification
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
    cv2.putText(blended, f"{device_label} | 104-COI | INFER: {latency_ms:.1f}ms ({fps:.1f} FPS)", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

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


def process_single_frame(frame_bgr: np.ndarray, source_name: str = "LAPTOP CAM -> PI INFER") -> Tuple[bytes, Dict[str, Any]]:
    """Runs full YOLIC TinyML pipeline on any frame and returns encoded JPEG + telemetry."""
    global global_model, global_smoother, global_polygons, global_transform

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = global_transform(T.functional.to_pil_image(rgb)).unsqueeze(0)

    with torch.no_grad():
        logits = global_model(tensor)
        raw_out = logits.squeeze(0).numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(raw_out, -20, 20)))

    probs = global_smoother.update(probs)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    rendered, active_hazards, actuation = render_yolic_hud(
        frame_bgr,
        probs,
        global_polygons,
        threshold=0.50,
        fps=fps,
        latency_ms=latency_ms,
        device_label=source_name
    )

    _, jpeg_bytes = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    telemetry = {
        "fps": round(fps, 1),
        "latency_ms": round(latency_ms, 1),
        "status": actuation["status_message"],
        "target_speed": actuation["target_speed"],
        "hazards_count": len(active_hazards),
        "active_hazards": active_hazards
    }

    return jpeg_bytes.tobytes(), telemetry


def pi_camera_worker(camera_index: int = 0):
    """Background worker for when Pi has its own camera connected."""
    global latest_jpeg, latest_telemetry

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Pi Camera index {camera_index} not detected. Ready for Laptop Camera streaming!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        jpeg_data, telemetry = process_single_frame(frame, source_name="PI CAMERA -> PI INFER")
        with frame_lock:
            latest_jpeg = jpeg_data
            latest_telemetry = telemetry


class StreamingHandler(BaseHTTPRequestHandler):
    """Serves the Web Dashboard, Laptop Camera receiver, and MJPEG video stream."""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/infer_laptop_frame":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            nparr = np.frombuffer(post_data, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                self.send_error(400, "Invalid image data")
                return

            # Process on Raspberry Pi CPU
            annotated_jpeg, telemetry = process_single_frame(frame_bgr, source_name="LAPTOP WEBCAM -> PI EDGE INFER")

            # Update latest telemetry
            with frame_lock:
                global latest_telemetry
                latest_telemetry = telemetry

            # Send back annotated JPEG + telemetry in Base64
            telemetry["image_b64"] = base64.b64encode(annotated_jpeg).decode("utf-8")
            response_bytes = json.dumps(telemetry).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
        else:
            self.send_error(404)

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
            text-align: center;
        }}
        .stream-img {{
            width: 100%;
            max-width: 640px;
            height: auto;
            border-radius: 6px;
            display: block;
            margin: 0 auto;
            background: #000;
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
        .btn {{
            background: #238636;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin-right: 10px;
        }}
        .btn-stop {{ background: #da3633; }}
        .btn:hover {{ opacity: 0.9; }}
        .cam-selector {{
            background: #21262d;
            color: white;
            border: 1px solid #30363d;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h2>🛴 Raspberry Pi Edge YOLIC Vision HUD</h2>
            <div style="color: #8b949e; font-size: 14px;">Extreme Edge TinyML Object Localization & Classification on Edge Devices (Paper: Su et al. 2024)</div>
        </div>
        <div style="text-align: right; font-size: 13px; color: #00ffaa;">
            🟢 RASPBERRY PI COMPUTE NODE
        </div>
    </div>

    <!-- Mode Selector Banner -->
    <div class="card" style="display: flex; justify-content: space-between; align-items: center; padding: 12px 18px;">
        <div>
            <strong>Camera Source:</strong>
            <label style="margin-left: 15px; cursor: pointer;">
                <input type="radio" name="cam_src" value="laptop" checked onchange="switchSource('laptop')"> 💻 Use Laptop's Camera (Webcam in Browser)
            </label>
            <label style="margin-left: 20px; cursor: pointer;">
                <input type="radio" name="cam_src" value="pi" onchange="switchSource('pi')"> 🍓 Use Pi's Camera (Plugged into Pi)
            </label>
        </div>
        <div id="laptop-controls">
            <button id="btn-start-cam" class="btn" onclick="startLaptopCamera()">▶️ Start Laptop Camera</button>
            <button id="btn-stop-cam" class="btn btn-stop" style="display: none;" onclick="stopLaptopCamera()">⏹️ Stop Camera</button>
        </div>
    </div>

    <div class="grid">
        <div class="stream-card">
            <img id="stream-display" class="stream-img" src="" alt="Stream View" />
            <video id="hidden-video" style="display: none;" playsinline></video>
            <canvas id="hidden-canvas" style="display: none;" width="640" height="480"></canvas>
            <div id="cam-status-msg" style="margin-top: 10px; color: #8b949e; font-size: 13px;">Click "Start Laptop Camera" to stream your laptop's camera directly to the Raspberry Pi for inference!</div>
        </div>

        <div>
            <div id="actuator-banner" class="actuator-box clear">ACTUATOR: ALL CLEAR (Cruising: 25 km/h)</div>

            <div class="card">
                <div class="metric-row">
                    <div class="metric">
                        <div id="metric-fps" class="metric-val">--</div>
                        <div class="metric-label">PI INFERENCE FPS</div>
                    </div>
                    <div class="metric">
                        <div id="metric-lat" class="metric-val">-- ms</div>
                        <div class="metric-label">EDGE LATENCY</div>
                    </div>
                    <div class="metric">
                        <div id="metric-speed" class="metric-val">25 km/h</div>
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
                        <tr><td colspan="4" style="color: #06d6a0;">✅ Path Clear. Zero hazards detected.</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentSource = 'laptop';
        let laptopStream = null;
        let isStreamingLaptop = false;
        let laptopInterval = null;

        const imgDisplay = document.getElementById('stream-display');
        const video = document.getElementById('hidden-video');
        const canvas = document.getElementById('hidden-canvas');
        const ctx = canvas.getContext('2d');
        const statusMsg = document.getElementById('cam-status-msg');

        function switchSource(src) {{
            currentSource = src;
            if (src === 'pi') {{
                stopLaptopCamera();
                document.getElementById('laptop-controls').style.display = 'none';
                imgDisplay.src = '/stream.mjpg';
                statusMsg.innerText = 'Streaming from camera connected directly to Raspberry Pi...';
            }} else {{
                document.getElementById('laptop-controls').style.display = 'block';
                imgDisplay.src = '';
                statusMsg.innerText = 'Click "Start Laptop Camera" to begin streaming your laptop camera to the Pi.';
            }}
        }}

        async function startLaptopCamera() {{
            try {{
                statusMsg.innerText = 'Accessing laptop webcam...';
                laptopStream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ width: 640, height: 480, facingMode: 'user' }}
                }});
                video.srcObject = laptopStream;
                await video.play();

                isStreamingLaptop = true;
                document.getElementById('btn-start-cam').style.display = 'none';
                document.getElementById('btn-stop-cam').style.display = 'inline-block';
                statusMsg.innerText = '🟢 Streaming laptop webcam frames to Raspberry Pi for real-time edge inference...';

                // Send frames to Raspberry Pi at ~20-25 FPS
                sendFrameLoop();
            }} catch (err) {{
                alert('Could not access laptop camera: ' + err.message);
                statusMsg.innerText = 'Error accessing camera: ' + err.message;
            }}
        }}

        function stopLaptopCamera() {{
            isStreamingLaptop = false;
            if (laptopStream) {{
                laptopStream.getTracks().forEach(track => track.stop());
                laptopStream = null;
            }}
            document.getElementById('btn-start-cam').style.display = 'inline-block';
            document.getElementById('btn-stop-cam').style.display = 'none';
            statusMsg.innerText = 'Camera stopped.';
        }}

        async function sendFrameLoop() {{
            if (!isStreamingLaptop) return;

            // Draw current video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert to JPEG blob
            canvas.toBlob(async (blob) => {{
                if (!blob || !isStreamingLaptop) return;

                const t0 = performance.now();
                try {{
                    const response = await fetch('/api/infer_laptop_frame', {{
                        method: 'POST',
                        body: blob
                    }});

                    if (response.ok) {{
                        const data = await response.json();
                        // Render returned annotated frame from Raspberry Pi
                        if (data.image_b64) {{
                            imgDisplay.src = 'data:image/jpeg;base64,' + data.image_b64;
                        }}
                        updateUI(data);
                    }}
                }} catch (e) {{
                    console.error(e);
                }}

                // Schedule next frame immediately after previous returns
                if (isStreamingLaptop) {{
                    setTimeout(sendFrameLoop, 15);
                }}
            }}, 'image/jpeg', 0.75);
        }}

        function updateUI(data) {{
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
                tbody.innerHTML = '<tr><td colspan="4" style="color: #06d6a0;">✅ All clear. Zero hazards detected.</td></tr>';
            }}
        }}

        // If in Pi camera mode, poll telemetry
        setInterval(async () => {{
            if (currentSource === 'pi') {{
                try {{
                    const res = await fetch('/api/telemetry');
                    const data = await res.json();
                    updateUI(data);
                }} catch (e) {{}}
            }}
        }}, 200);
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
    parser.add_argument("--camera", type=int, default=0, help="Pi Camera device index if connected (default: 0)")
    args = parser.parse_args()

    # 1. Initialize YOLIC Pipeline on Raspberry Pi
    init_yolic_pipeline()

    # 2. Start Pi camera thread (if Pi camera is plugged in)
    pi_cam_thread = Thread(target=pi_camera_worker, args=(args.camera,), daemon=True)
    pi_cam_thread.start()

    local_ip = get_local_ip()
    server_address = ("0.0.0.0", args.port)
    httpd = HTTPServer(server_address, StreamingHandler)

    print("\n" + "=" * 70)
    print("🚀 RASPBERRY PI YOLIC EDGE SERVER RUNNING")
    print("=" * 70)
    print(f"📡 Pi Local Address:   http://localhost:{args.port}")
    print(f"💻 Open on your Laptop: http://{local_ip}:{args.port}")
    print("   👉 Click 'Start Laptop Camera' on your laptop to use your laptop webcam!")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Edge Server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
