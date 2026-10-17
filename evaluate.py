"""
Evaluation Protocol for Extended YOLIC
Metrics computed:
1. Per-class Precision, Recall, F1 for all M target classes.
2. Binary Risk-vs-Road Precision, Recall, F1.
3. Temporal Flicker Rate across video frames (demonstrating TemporalSmoother flicker reduction).
"""
from typing import Dict, List, Tuple
import argparse
import os
import yaml
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.unified_yolic import UnifiedYOLIC, build_yolic_from_config
from models.temporal_smoother import TemporalSmoother
from datasets.coi_dataset import get_dataloader
from datasets.synthetic_generator import SyntheticSceneGenerator


def compute_per_class_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    class_names: List[str]
) -> List[Dict[str, float]]:
    """
    Computes Precision, Recall, and F1 per class.
    Args:
        all_preds: binary (Total_Cells, Total_Classes)
        all_targets: binary (Total_Cells, Total_Classes)
    """
    num_classes = len(class_names)
    results = []

    for c in range(num_classes):
        p = all_preds[:, c]
        t = all_targets[:, c]

        tp = np.sum((p == 1) & (t == 1))
        fp = np.sum((p == 1) & (t == 0))
        fn = np.sum((p == 0) & (t == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        results.append({
            "class_id": c,
            "class_name": class_names[c],
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(t))
        })
    return results


def compute_road_risk_metrics(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    bg_idx: int,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Binary Risk-vs-Road evaluation:
    - Cell is Road if P_bg >= 0.5
    - Cell is Risk if P_bg < 0.5
    """
    pred_road = (all_probs[:, bg_idx] >= threshold).astype(np.int32)
    true_road = all_targets[:, bg_idx].astype(np.int32)

    # Road metrics
    tp_road = np.sum((pred_road == 1) & (true_road == 1))
    fp_road = np.sum((pred_road == 1) & (true_road == 0))
    fn_road = np.sum((pred_road == 0) & (true_road == 1))

    prec_road = tp_road / (tp_road + fp_road + 1e-8)
    rec_road = tp_road / (tp_road + fn_road + 1e-8)
    f1_road = 2 * (prec_road * rec_road) / (prec_road + rec_road + 1e-8)

    # Risk metrics (Risk = 1 - Road)
    pred_risk = 1 - pred_road
    true_risk = 1 - true_road
    tp_risk = np.sum((pred_risk == 1) & (true_risk == 1))
    fp_risk = np.sum((pred_risk == 1) & (true_risk == 0))
    fn_risk = np.sum((pred_risk == 0) & (true_risk == 1))

    prec_risk = tp_risk / (tp_risk + fp_risk + 1e-8)
    rec_risk = tp_risk / (tp_risk + fn_risk + 1e-8)
    f1_risk = 2 * (prec_risk * rec_risk) / (prec_risk + rec_risk + 1e-8)

    return {
        "road_precision": float(prec_road),
        "road_recall": float(rec_road),
        "road_f1": float(f1_road),
        "risk_precision": float(prec_risk),
        "risk_recall": float(rec_risk),
        "risk_f1": float(f1_risk),
        "road_support": int(np.sum(true_road)),
        "risk_support": int(np.sum(true_risk))
    }


def compute_temporal_flicker_rate(
    model: torch.nn.Module,
    device: torch.device,
    num_frames: int = 40,
    alpha: float = 0.8,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes Temporal Flicker Rate:
    Flicker = (1 / ((T - 1) * N * (M + 1))) * sum_{t=1}^{T-1} | Y_{t+1} - Y_t |
    Compares raw predictions vs. smoothed predictions to measure flicker suppression.
    """
    synth = SyntheticSceneGenerator(num_cois=104, num_classes=11)
    video_seq = synth.generate_video_sequence(num_frames=num_frames)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    smoother = TemporalSmoother(alpha=alpha)
    smoother.reset()

    raw_preds_seq = []
    smoothed_preds_seq = []

    model.eval()
    with torch.no_grad():
        for img_np, _ in video_seq:
            img_pil = Image.fromarray(img_np)
            tensor = transform(img_pil).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0]  # (N, M + 1)

            # Raw prediction
            raw_binary = (probs >= threshold).float().cpu().numpy()
            raw_preds_seq.append(raw_binary)

            # Smoothed prediction
            smooth_p = smoother.update(probs)
            smooth_binary = (smooth_p >= threshold).float().cpu().numpy()
            smoothed_preds_seq.append(smooth_binary)

    T_count = len(video_seq)
    N = raw_preds_seq[0].shape[0]
    total_classes = raw_preds_seq[0].shape[1]
    denom = (T_count - 1) * N * total_classes

    # Compute raw flicker
    raw_diff = sum(np.sum(np.abs(raw_preds_seq[t+1] - raw_preds_seq[t])) for t in range(T_count - 1))
    raw_flicker = raw_diff / denom

    # Compute smoothed flicker
    smooth_diff = sum(np.sum(np.abs(smoothed_preds_seq[t+1] - smoothed_preds_seq[t])) for t in range(T_count - 1))
    smooth_flicker = smooth_diff / denom

    improvement_pct = max(0.0, (raw_flicker - smooth_flicker) / (raw_flicker + 1e-8) * 100.0)
    return float(raw_flicker), float(smooth_flicker), float(improvement_pct)


def evaluate(args):
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"Evaluation Device: {device}")

    # Build model
    if cfg:
        model = build_yolic_from_config(cfg, pretrained=False).to(device)
        class_names = cfg.get("dataset", {}).get("class_names", [])
        num_cois = cfg.get("dataset", {}).get("num_cois", 104)
        bg_idx = cfg.get("dataset", {}).get("background_index", 11)
    else:
        model = UnifiedYOLIC(num_cois=104, num_classes=11, backbone_type="shufflenet_v2", pretrained_backbone=False).to(device)
        class_names = [
            "Pothole", "Speed Bump", "Manhole Cover", "Pedestrian",
            "Bicycle", "Car", "Debris", "Curb", "Stop Sign",
            "Yield Sign", "Warning Sign"
        ]
        num_cois = 104
        bg_idx = 11

    # Load weights if available
    checkpoint_path = args.checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
    else:
        print("Notice: No trained checkpoint specified or found. Evaluating model directly.")

    model.eval()

    # Dataloader for multi-sample test
    loader = get_dataloader(cfg, split="val", batch_size=16, synthetic=args.synthetic, num_samples=args.samples, shuffle=False)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)      # (B_total, N, M + 1)
    all_targets = np.concatenate(all_targets, axis=0)  # (B_total, N, M + 1)

    # Flatten across batch and CoIs: (B_total * N, M + 1)
    flat_probs = all_probs.reshape(-1, all_probs.shape[-1])
    flat_targets = all_targets.reshape(-1, all_targets.shape[-1])
    flat_preds = (flat_probs >= args.threshold).astype(np.int32)

    # 1. Per-Class F1 Score
    per_class = compute_per_class_metrics(flat_preds[:, :bg_idx], flat_targets[:, :bg_idx], class_names)

    # 2. Road vs Risk F1
    road_risk = compute_road_risk_metrics(flat_probs, flat_targets, bg_idx=bg_idx, threshold=args.threshold)

    # 3. Temporal Flicker Rate
    raw_fl, sm_fl, fl_imp = compute_temporal_flicker_rate(model, device, num_frames=30, alpha=args.alpha, threshold=args.threshold)

    # Display Results
    print("\n" + "=" * 76)
    print("                      YOLIC EVALUATION RESULTS                      ")
    print("=" * 76)

    print("\n--- 1. Per-Class Hazard Metrics (Threshold tau = 0.5) ---")
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 76)
    f1_list = []
    for item in per_class:
        f1_list.append(item['f1'])
        print(f"{item['class_name']:<22} | {item['precision']:<10.4f} | {item['recall']:<10.4f} | {item['f1']:<10.4f} | {item['support']:<8d}")
    print("-" * 76)
    print(f"{'Macro Average F1':<22} | {'-':<10} | {'-':<10} | {np.mean(f1_list):<10.4f} | {int(np.sum(flat_targets[:, :bg_idx])):<8d}")

    print("\n--- 2. Binary Risk-vs-Road Detection ---")
    print(f"Clear Road Cell F1: {road_risk['road_f1']:.4f}  (Precision: {road_risk['road_precision']:.4f}, Recall: {road_risk['road_recall']:.4f})")
    print(f"Hazard Risk Cell F1: {road_risk['risk_f1']:.4f}  (Precision: {road_risk['risk_precision']:.4f}, Recall: {road_risk['risk_recall']:.4f})")

    print("\n--- 3. Temporal Consistency & Prediction Flicker Rate ---")
    print(f"Raw Prediction Flicker Rate:      {raw_fl:.6f}")
    print(f"Smoothed Prediction Flicker Rate: {sm_fl:.6f}  (Alpha = {args.alpha})")
    print(f"Flicker Suppression Gain:         {fl_imp:.2f}% reduction")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended YOLIC Evaluation Script")
    parser.add_argument("--config", type=str, default="configs/outdoor_104coi.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/outdoor_104coi/best_model.pth")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(args)
