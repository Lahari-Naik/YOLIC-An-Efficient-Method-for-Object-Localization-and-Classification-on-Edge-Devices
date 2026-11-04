"""
Download Pretrained YOLIC ONNX Weights from the Paper Author's Repository
"""

import os
import urllib.request
import shutil

ONNX_URL = "https://raw.githubusercontent.com/kai3316/YOLIC-Labeling/master/oxxn%20models%20for%20config3/Config3_resnet50_rgb%20mode.onnx"
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(TARGET_DIR, "Config3_resnet50_rgb.onnx")

def download_weights():
    if os.path.exists(TARGET_PATH):
        print(f"Weights already exist at: {TARGET_PATH}")
        return

    print(f"Downloading pre-trained weights from {ONNX_URL} ...")
    urllib.request.urlretrieve(ONNX_URL, TARGET_PATH)
    print(f"Successfully downloaded to: {TARGET_PATH}")

if __name__ == "__main__":
    download_weights()
