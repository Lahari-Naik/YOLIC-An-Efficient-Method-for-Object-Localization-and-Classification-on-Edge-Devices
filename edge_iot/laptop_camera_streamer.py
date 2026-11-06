"""
Laptop Camera Streamer for YOLIC Edge Inference
Captures video from the laptop's built-in webcam via OpenCV, streams frames
over Wi-Fi to the Raspberry Pi for TinyML processing, and displays the returned
annotated HUD in an OpenCV desktop window.
"""

import argparse
import time
import base64
import json
import cv2
import requests

def main():
    parser = argparse.ArgumentParser(description="Laptop Camera Streamer to Raspberry Pi YOLIC")
    parser.add_argument("--pi-ip", type=str, required=True, help="IP address of Raspberry Pi (e.g. 192.168.1.50)")
    parser.add_argument("--port", type=int, default=8080, help="Server port on Raspberry Pi (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Laptop webcam index (default: 0)")
    args = parser.parse_args()

    api_url = f"http://{args.pi_ip}:{args.port}/api/infer_laptop_frame"
    print(f"Connecting laptop webcam {args.camera} to Raspberry Pi at: {api_url}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open laptop webcam index {args.camera}.")
        return

    # Set 640x480 resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = f"YOLIC HUD (Laptop Webcam -> Pi Inference @ {args.pi_ip})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nStreaming live video to Raspberry Pi... Press 'q' or 'ESC' in the window to quit.\n")

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from laptop webcam.")
            break

        # Encode frame to JPEG
        _, jpeg_encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        jpeg_bytes = jpeg_encoded.tobytes()

        # Send to Raspberry Pi over HTTP POST
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                api_url,
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=2.0
            )

            if resp.status_code == 200:
                data = resp.json()
                t1 = time.perf_counter()
                roundtrip_ms = (t1 - t0) * 1000.0

                # Decode the annotated HUD frame returned by Raspberry Pi
                img_b64 = data.get("image_b64")
                if img_b64:
                    raw_returned = base64.b64decode(img_b64)
                    nparr = cv2.imdecode(cv2.frombuffer(raw_returned, dtype=cv2.CV_8U), cv2.IMREAD_COLOR)

                    # Add Roundtrip Latency tag
                    cv2.putText(
                        nparr,
                        f"ROUNDTRIP (WIFI+PI): {roundtrip_ms:.1f}ms",
                        (14, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    cv2.imshow(window_name, nparr)
            else:
                print(f"Server returned status {resp.status_code}")

        except Exception as e:
            # Show original frame with warning overlay
            cv2.putText(
                frame,
                f"Waiting for Raspberry Pi at {args.pi_ip}...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended cleanly.")

if __name__ == "__main__":
    main()
