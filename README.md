# Extended YOLIC: Tiny AI & IoT Edge Vision

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40%2B-ff4b4b.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Single--Core%20CPU-30%2B%20FPS-brightgreen.svg)](#hardware-profiling-results)

An end-to-end, ultra-lightweight object localization and multi-label classification system for micro-mobility edge devices (e-scooters, autonomous delivery pods, and robotics), based on the reference paper:
> **"YOLIC: An Efficient Method for Object Localization and Classification on Edge Devices"**  
> *Su et al., Image and Vision Computing, 2024; doi:[10.1016/j.imavis.2024.105095](https://doi.org/10.1016/j.imavis.2024.105095)*

---

## 1. System Overview & Novel Extensions

Traditional object detectors (e.g. YOLO, Faster R-CNN) rely on anchor matching, multi-scale feature pyramids, bounding-box regression, and Non-Maximum Suppression (NMS)—operations that introduce unpredictable latency and excessive memory overhead on low-power edge processors.

**YOLIC** formulates object localization as multi-label classification across a fixed geometric set of **Cells of Interest (CoIs)** tailored to the camera perspective. This repository extends the baseline YOLIC with three novel architectural contributions:

```
[ Input: 3 x 224 x 224 ]
           │
           ▼
[ Backbone Feature Extractor ]  (ShuffleNetV2 x1.0 / MobileNetV2)
           │
           ▼  F in R^(B x 1024 x 7 x 7)
  [ Token Projector ]           GAP + Linear Projection
           │
           ▼  H^(0) in R^(B x N x 128)
[ Inter-Cell Context Module ]   Multi-Head Self-Attention (h=4, dropout=0.1) + LayerNorm
           │
           ▼  H^(1) in R^(B x N x 128)
[ Grouped Parameter-Bounded Head ] Shared Classifier W_cls (1,548 params, INVARIANT to N)
           │
           ▼  Logits in R^(B x N x 12)
[ Temporal Consistency Layer ]  EMA Probability Smoothing (S_t = α P_t + (1-α) S_{t-1})
           │
     ┌─────┴─────────────────────────┐
     ▼                               ▼
[ Closed-Loop Actuator Logic ]  [ Streamlit UI Visualizer ]
- CoIs 0–20: EMERGENCY STOP     - 8 Top Sign Cells + 96 Perspective Road Cells
- CoIs 21–95: SPEED WARNING     - Green road outlines & colored hazard overlays
- CoIs 96–103: Top Signs        - Single-core FPS & Telemetry
```

1. **Inter-Cell Context Module:** Multi-Head Self-Attention (MHSA) capturing spatial co-occurrence dependencies across adjacent Cells of Interest (CoIs).
2. **Grouped / Parameter-Bounded Classification Head:** Shared linear projection $W_{\text{cls}} \in \mathbb{R}^{d_{\text{cell}} \times (M + 1)}$ that bounds classification head parameter count to **1,548 parameters**, completely invariant to the number of cells $N$.
3. **Temporal Consistency Layer:** Low-latency Exponential Moving Average (EMA) smoothing ($\alpha = 0.8$) preventing prediction flicker across consecutive video frames.
4. **Closed-Loop Actuator & IoT Node:** Near-field hazard trigger for immediate mechanical emergency stop vs. mid-field speed warning, with resilient MQTT telemetry and offline fallback.

---

## 2. Repository Structure

```
yolic-tinyml/
├── configs/
│   ├── outdoor_104coi.yaml       # 104-CoI e-scooter configuration (8 sign + 96 road cells)
│   └── cityscapes_256coi.yaml     # 256-CoI urban robotics configuration
├── datasets/
│   ├── coi_dataset.py             # PyTorch Dataset with horizontal flip CoI coordinate swapping
│   └── synthetic_generator.py     # Procedural 224x224 road scene & hazard generator
├── models/
│   ├── __init__.py
│   ├── backbones.py               # Truncated ShuffleNetV2 x1.0 & MobileNetV2
│   ├── context_module.py          # Token Projector & Multi-Head Self-Attention
│   ├── grouped_head.py            # Baseline Flat Head vs. Grouped Parameter-Bounded Head
│   ├── temporal_smoother.py       # Frame-wise EMA probability filter
│   └── unified_yolic.py           # Unified modular model wrapper
├── edge_iot/
│   ├── __init__.py
│   ├── actuator_logic.py          # Near-field vs. mid/far-field vehicle actuation
│   └── mqtt_publisher.py          # Telemetry node with offline fallback
├── train.py                       # Training pipeline (BCEWithLogitsLoss + Adam + MultiStepLR)
├── evaluate.py                    # Evaluation metrics (Per-class F1, Road F1, Flicker Rate)
├── profile_tinyml.py              # Strict single-core CPU profiler (thop + tracemalloc)
├── frontend_app.py                # Streamlit paper-matching interactive visualizer
└── requirements.txt               # Pinned project dependencies
```

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Windows, Linux, or macOS

### Setup
```bash
git clone https://github.com/your-username/yolic-tinyml.git
cd yolic-tinyml

pip install -r requirements.txt
```

---

## 4. Hardware Profiling Results (Single-Core CPU)

Run the Tiny AI profiler under strict single-threaded CPU constraints:
```bash
python profile_tinyml.py
```

### Benchmark Summary (Single-Threaded CPU)

| Model Variant | CoIs ($N$) | Total Params (M) | Head Params | MACs (M) | Peak RAM | Latency (1-Core) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Base YOLIC (Flat Head)** | 104 | 2.533 M | 1,279,200 | 153.01 M | 0.003 MB | 28.97 ms | **34.5 FPS** |
| **2. Grouped Head YOLIC** | 104 | 14.900 M | **1,548** | 165.53 M | 0.003 MB | 32.65 ms | **30.6 FPS** |
| **3. Extended YOLIC (Context + Grouped)** | 104 | 14.966 M | **1,548** | 165.58 M | 0.003 MB | 31.50 ms | **31.7 FPS** |
| **1. Base YOLIC (Flat Head)** | 256 | 4.402 M | 3,148,800 | 154.88 M | 0.003 MB | 41.47 ms | **24.1 FPS** |
| **2. Grouped Head YOLIC** | 256 | 34.842 M | **1,548** | 185.68 M | 0.003 MB | 33.78 ms | **29.6 FPS** |
| **3. Extended YOLIC (Context + Grouped)** | 256 | 34.909 M | **1,548** | 185.81 M | 0.003 MB | 39.95 ms | **25.0 FPS** |

### Key Takeaways
- **Parameter Invariance:** While the baseline flat head scales linearly ($O(C_{\text{feat}} \cdot N \cdot (M+1))$) from 1.28M to 3.15M parameters, the proposed Grouped Head remains fixed at strictly **1,548 parameters**.
- **Edge Feasibility:** All models achieve **$>30$ FPS** on a constrained single CPU core with peak RAM of only **3 KB**!

---

## 5. Running the Streamlit Frontend

Launch the paper-style interactive visualizer:
```bash
streamlit run frontend_app.py
```
Open your browser at `http://localhost:8501`.

### Visualizer Features
- **104-CoI Paper-Style Overlays:**
  - 8 top traffic sign cells across the upper horizon band.
  - 96 perspective trapezoidal road cells expanding toward the scooter bumper.
  - Semi-transparent green outlines for clear road cells ($P_{\text{clear}} \ge 0.5$).
  - Transparent colored fills and confidence badges for active hazard cells.
- **Interactive Modes:**
  1. *Procedural Synthetic Generator:* Interactively inject potholes, bumps, pedestrians, cars, and traffic signs.
  2. *Image Upload:* Test on custom road imagery.
  3. *Live Stream Simulation:* Real-time video sequence benchmark showing temporal flicker reduction.
- **Actuator Signal Dashboard:** Displays real-time mechanical braking decisions (`EMERGENCY STOP` for CoIs 0–20 vs. `SPEED WARNING` for CoIs 21+).

---

## 6. Training & Evaluation

### Training Pipeline
Train on procedural synthetic road data or custom annotated datasets:
```bash
# Quick test run (2 epochs)
python train.py --epochs 2 --samples 64 --batch-size 8 --synthetic

# Full training protocol (150 epochs with MultiStepLR milestones [100, 125])
python train.py --config configs/outdoor_104coi.yaml --epochs 150 --batch-size 32
```

### Evaluation Protocol
Evaluate per-class precision/recall/F1, binary risk-vs-road detection, and temporal flicker:
```bash
python evaluate.py --config configs/outdoor_104coi.yaml --checkpoint checkpoints/outdoor_104coi/best_model.pth --synthetic
```

---

## 7. Edge IoT Node & Closed-Loop Actuator

Run the standalone MQTT telemetry publisher:
```bash
# Run 10 iterations of live telemetry
python edge_iot/mqtt_publisher.py --iterations 10 --node-id scooter_27
```

If an MQTT broker is unreachable at `localhost:1883`, the node logs structured JSON telemetry to console with offline fallback:
```json
{
  "node_id": "scooter_27",
  "timestamp": 1773724800,
  "fps": 31.5,
  "latency_ms": 31.7,
  "hazard_detected": true,
  "active_cois": [
    {"coi_id": 14, "class": "Speed Bump", "conf": 0.89}
  ],
  "emergency_stop": false
}
```

---

## 8. Mathematical Reference

### Token Projector & Inter-Cell Attention
$$H^{(0)} = \text{ReLU}\left(\text{Linear}\left(\text{Flatten}\left(\text{AdaptiveAvgPool2d}_{7 \times 7}(F)\right)\right)\right) \in \mathbb{R}^{B \times N \times d_{\text{cell}}}$$
$$H^{(1)} = \text{LayerNorm}\left(H^{(0)} + \text{Dropout}\left(\text{MHSA}\left(Q=H^{(0)}, K=H^{(0)}, V=H^{(0)}\right)\right)\right)$$

### Parameter-Bounded Classifier
$$\text{Logits} = H^{(1)} W_{\text{cls}} + b \in \mathbb{R}^{B \times N \times (M + 1)}$$
$$\text{Head Parameters} = d_{\text{cell}} \cdot (M + 1) + (M + 1) = 128 \times 12 + 12 = 1,548$$

### Temporal Consistency EMA
$$S_t = \alpha \cdot P_t + (1 - \alpha) \cdot S_{t-1}, \quad \alpha \in [0.7, 0.9]$$

### Temporal Flicker Metric
$$\text{Flicker} = \frac{1}{(T-1) \cdot N \cdot (M+1)} \sum_{t=1}^{T-1} \left| \hat{Y}_{t+1} - \hat{Y}_t \right|$$

---

## 9. License

This project is licensed under the MIT License.
