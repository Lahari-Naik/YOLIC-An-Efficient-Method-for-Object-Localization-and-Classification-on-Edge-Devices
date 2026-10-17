"""
IoT Telemetry Node & MQTT Publisher for YOLIC
Publishes structured JSON telemetry to MQTT broker with graceful offline fallback.
"""
from typing import Any, Dict, List, Optional
import argparse
import json
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from models.unified_yolic import UnifiedYOLIC
from models.temporal_smoother import TemporalSmoother
from datasets.synthetic_generator import SyntheticSceneGenerator

try:
    from edge_iot.actuator_logic import ActuatorLogic
except ImportError:
    from actuator_logic import ActuatorLogic


class MQTTPublisher:
    """
    Publishes live vision inference telemetry to an MQTT broker.
    Handles network disconnections with seamless offline fallback.
    """
    def __init__(
        self,
        node_id: str = "scooter_27",
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "iot/scooter/telemetry"
    ):
        self.node_id = node_id
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client: Optional[Any] = None
        self.is_connected = False
        self._init_client()

    def _init_client(self) -> None:
        if not PAHO_AVAILABLE:
            print("[MQTTPublisher] Notice: paho-mqtt not available. Running in offline fallback mode.")
            return

        try:
            # Support both paho-mqtt v1 and v2 API
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.node_id)
            else:
                self.client = mqtt.Client(client_id=self.node_id)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            print(f"[MQTTPublisher] Connecting to MQTT broker {self.broker}:{self.port}...")
            self.client.connect_async(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTTPublisher] MQTT Connection failed ({e}). Offline fallback mode active.")
            self.client = None
            self.is_connected = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            print(f"[MQTTPublisher] Connected successfully to {self.broker}:{self.port} on topic '{self.topic}'.")
        else:
            self.is_connected = False
            print(f"[MQTTPublisher] Connection refused with code: {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self.is_connected = False
        print("[MQTTPublisher] Disconnected from MQTT broker.")

    def publish_telemetry(self, telemetry_data: Dict[str, Any]) -> bool:
        """
        Publishes telemetry dictionary as JSON.
        Returns True if sent to network broker, False if logged offline.
        """
        payload = json.dumps(telemetry_data)
        if self.client is not None and self.is_connected:
            try:
                self.client.publish(self.topic, payload, qos=0)
                return True
            except Exception as e:
                print(f"[MQTTPublisher] Error publishing to MQTT ({e}). Falling back to offline log.")

        # Offline fallback: Print telemetry packet
        print(f"[OFFLINE TELEMETRY] -> {payload}")
        return False

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass


def run_iot_loop(
    model_path: Optional[str] = None,
    broker: str = "localhost",
    port: int = 1883,
    topic: str = "iot/scooter/telemetry",
    node_id: str = "scooter_27",
    max_iterations: int = 10,
    synthetic: bool = True
):
    """Runs continuous inference and telemetry loop."""
    print("==================================================")
    print("      YOLIC IoT Telemetry & Actuator Node         ")
    print("==================================================")

    # Initialize model
    device = torch.device("cpu")
    model = UnifiedYOLIC(
        num_cois=104,
        num_classes=11,
        backbone_type="shufflenet_v2",
        use_context_module=True,
        use_grouped_head=True,
        pretrained_backbone=False
    ).to(device)
    model.eval()

    if model_path and os.path.exists(model_path):
        print(f"Loading checkpoint from: {model_path}")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))

    smoother = TemporalSmoother(alpha=0.8)
    actuator = ActuatorLogic(near_field_max_idx=20)
    publisher = MQTTPublisher(node_id=node_id, broker=broker, port=port, topic=topic)
    synth_gen = SyntheticSceneGenerator(num_cois=104, num_classes=11)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = [
        "Pothole", "Speed Bump", "Manhole Cover", "Pedestrian",
        "Bicycle", "Car", "Debris", "Curb", "Stop Sign",
        "Yield Sign", "Warning Sign"
    ]

    iteration = 0
    try:
        while max_iterations <= 0 or iteration < max_iterations:
            iteration += 1
            t_start = time.perf_counter()

            # Generate or grab frame
            img_np, _, _ = synth_gen.generate_frame()
            img_pil = Image.fromarray(img_np)
            tensor = transform(img_pil).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)[0]  # (104, 12)

            # Temporal smoothing
            smoothed_probs = smoother.update(probs)
            t_end = time.perf_counter()

            latency_ms = (t_end - t_start) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # Detect active hazard CoIs
            # Paper definition: Clear road if P_bg >= 0.5; Hazard if P_bg < 0.5 and max hazard prob >= 0.5
            bg_prob = smoothed_probs[:, 11]
            hazard_probs = smoothed_probs[:, :11]
            max_hazard_probs, best_classes = torch.max(hazard_probs, dim=1)

            active_cois = []
            for coi_idx in range(104):
                if bg_prob[coi_idx] < 0.5 and max_hazard_probs[coi_idx] >= 0.5:
                    cls_id = int(best_classes[coi_idx].item())
                    conf = float(max_hazard_probs[coi_idx].item())
                    active_cois.append({
                        "coi_id": coi_idx,
                        "class": class_names[cls_id],
                        "conf": round(conf, 2)
                    })

            # Actuator closed-loop logic
            actuation = actuator.evaluate(active_cois)

            # JSON Telemetry payload
            telemetry = {
                "node_id": node_id,
                "timestamp": int(time.time()),
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "hazard_detected": actuation["hazard_detected"],
                "active_cois": active_cois,
                "emergency_stop": actuation["emergency_stop"]
            }

            publisher.publish_telemetry(telemetry)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[MQTTPublisher] Interrupted by user.")
    finally:
        publisher.close()
        print("[MQTTPublisher] IoT Telemetry Loop shutdown cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLIC IoT Telemetry Node")
    parser.add_argument("--broker", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", type=str, default="iot/scooter/telemetry")
    parser.add_argument("--node-id", type=str, default="scooter_27")
    parser.add_argument("--iterations", type=int, default=5, help="Number of telemetry cycles (0 for infinite)")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    run_iot_loop(
        model_path=args.model,
        broker=args.broker,
        port=args.port,
        topic=args.topic,
        node_id=args.node_id,
        max_iterations=args.iterations
    )
