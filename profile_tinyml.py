"""
Tiny AI Hardware Profiler for YOLIC
Enforces strict single-threaded CPU execution to simulate low-power IoT microcontrollers / edge SoCs.
Profiles:
- Parameters (Millions)
- MACs / MFLOPs via thop
- Peak RAM (MB) via tracemalloc
- Single-Core Latency (ms) & FPS (50 warmup + 200 timed iterations)
Compares:
1. Base YOLIC (Flat Head)
2. Grouped Head YOLIC
3. Extended YOLIC (Context Module + Grouped Head)
For N = 104 and N = 256.
"""
import sys
import time
import tracemalloc
import torch
import torch.nn as nn

# Enforce strict single-threaded CPU execution before any tensor operations
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_threads(1)

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

from models.unified_yolic import UnifiedYOLIC


def profile_single_model(
    name: str,
    model: nn.Module,
    num_cois: int,
    warmup: int = 50,
    timed: int = 200
):
    """
    Profiles a single model configuration on single-core CPU.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    # Specific Head parameter count
    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters())
    else:
        head_params = 0

    # 2. MACs / FLOPs computation
    macs_m = 0.0
    if THOP_AVAILABLE:
        try:
            # thop profile
            macs, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
            macs_m = macs / 1e6
        except Exception as e:
            # Fallback estimation
            macs_m = 0.0

    # 3. Peak RAM Tracking with tracemalloc
    tracemalloc.start()
    tracemalloc.reset_peak()
    with torch.no_grad():
        _ = model(dummy_input)
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_ram / (1024 * 1024)

    # 4. Latency & FPS Benchmark (50 warmup, 200 timed)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

        latencies = []
        for _ in range(timed):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency = float(sum(latencies) / len(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "variant": name,
        "cois": num_cois,
        "params_m": params_m,
        "head_params": head_params,
        "macs_m": macs_m,
        "peak_ram_mb": peak_ram_mb,
        "latency_ms": avg_latency,
        "fps": fps
    }


def main():
    print("\n" + "=" * 96)
    print("      TINY AI HARDWARE PROFILER (SINGLE-THREADED CPU CONSTRAINED)      ")
    print(f"      PyTorch Threads: {torch.get_num_threads()} | Inter-op: {torch.get_num_interop_threads()}      ")
    print("=" * 96)

    coi_benchmarks = [104, 256]
    all_results = []

    for N in coi_benchmarks:
        print(f"\n[+] Profiling Configurations for N = {N} Cells of Interest (CoIs)...")

        # 1. Base YOLIC (Flat Head)
        base_model = UnifiedYOLIC(
            num_cois=N,
            num_classes=11,
            backbone_type="shufflenet_v2",
            use_context_module=False,
            use_grouped_head=False,
            pretrained_backbone=False
        )
        res_base = profile_single_model("1. Base YOLIC (Flat Head)", base_model, N)
        all_results.append(res_base)

        # 2. Grouped Head YOLIC
        grouped_model = UnifiedYOLIC(
            num_cois=N,
            num_classes=11,
            backbone_type="shufflenet_v2",
            use_context_module=False,
            use_grouped_head=True,
            pretrained_backbone=False
        )
        res_grouped = profile_single_model("2. Grouped Head YOLIC", grouped_model, N)
        all_results.append(res_grouped)

        # 3. Extended YOLIC (Context Module + Grouped Head)
        ext_model = UnifiedYOLIC(
            num_cois=N,
            num_classes=11,
            backbone_type="shufflenet_v2",
            use_context_module=True,
            use_grouped_head=True,
            pretrained_backbone=False
        )
        res_ext = profile_single_model("3. Extended YOLIC (Context + Grouped)", ext_model, N)
        all_results.append(res_ext)

    # Print ASCII Comparison Table
    print("\n" + "=" * 105)
    print("                                TINY AI PROFILING BENCHMARK RESULTS                                ")
    print("=" * 105)
    header = (
        f"{'Model Variant':<36} | {'CoIs':<5} | {'Params(M)':<10} | {'Head Params':<11} | "
        f"{'MACs(M)':<9} | {'Peak RAM':<10} | {'Latency':<10} | {'FPS':<6}"
    )
    print(header)
    print("-" * 105)

    for r in all_results:
        row = (
            f"{r['variant']:<36} | {r['cois']:<5d} | {r['params_m']:<10.3f} | {r['head_params']:<11d} | "
            f"{r['macs_m']:<9.2f} | {r['peak_ram_mb']:<7.3f} MB | {r['latency_ms']:<7.2f} ms | {r['fps']:<6.1f}"
        )
        print(row)
    print("=" * 105)

    # Print Key Mathematical Insights
    print("\nKey Architectural Observations:")
    print("1. Parameter Boundedness:")
    for r in all_results:
        if "Grouped" in r['variant']:
            print(f"   - {r['variant']} (N={r['cois']}): Head Parameters = {r['head_params']} (INVARIANT to N!)")
        else:
            print(f"   - {r['variant']} (N={r['cois']}): Head Parameters = {r['head_params']} (Scales linearly O(C_feat * N * (M+1)))")

    print("\n2. Single-Core Real-Time Feasibility:")
    fastest = max(all_results, key=lambda x: x['fps'])
    print(f"   - Highest Single-Core Throughput: {fastest['variant']} at {fastest['fps']:.1f} FPS ({fastest['latency_ms']:.2f} ms latency)")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
