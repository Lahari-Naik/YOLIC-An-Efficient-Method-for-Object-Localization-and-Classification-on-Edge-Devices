#!/bin/bash
# ==============================================================================
# One-Click Raspberry Pi Setup Script for YOLIC Edge Vision
# Supports Raspberry Pi 4 / 5 / 3B+ (Raspberry Pi OS 32/64-bit)
# ==============================================================================

set -e

echo "=================================================="
echo "🍓 Setting up YOLIC Edge Vision on Raspberry Pi..."
echo "=================================================="

# 1. System dependencies
echo "📦 Updating apt packages..."
sudo apt update
sudo apt install -y python3-pip python3-opencv python3-numpy libatlas-base-dev libopenjp2-7 libtiff5

# 2. PyTorch CPU installation for ARM (Optimized wheel)
echo "⚡ Installing PyTorch (CPU) for Raspberry Pi..."
pip3 install --upgrade pip
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Python dependencies
echo "📚 Installing project requirements..."
pip3 install paho-mqtt pyyaml pillow thop

# 4. Permissions for camera
echo "📷 Configuring camera permissions..."
sudo usermod -aG video $USER || true

# 5. Display IP Address
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "=================================================="
echo "✅ SETUP COMPLETE!"
echo "=================================================="
echo "Your Raspberry Pi IP Address is: $LOCAL_IP"
echo ""
echo "👉 To start the ultra-lightweight Edge Server:"
echo "   python3 edge_iot/pi_edge_server.py --port 8080"
echo ""
echo "💻 Then open your laptop browser and visit:"
echo "   http://$LOCAL_IP:8080"
echo "=================================================="
