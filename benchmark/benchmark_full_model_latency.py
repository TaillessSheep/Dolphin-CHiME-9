#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def default_dolphin_root() -> Path:
    """
    Expected layout:

    capstone/
    ├── Dolphin/
    │   └── Dolphin/
    │       ├── Inference.py
    │       └── look2hear/
    └── Dolphin-CHiME-9/
        └── benchmark/
            └── benchmark_full_model_latency.py

    This script is under Dolphin-CHiME-9/benchmark/.
    So:
      script_path.parents[0] = benchmark
      script_path.parents[1] = Dolphin-CHiME-9
      script_path.parents[2] = capstone
    """
    script_path = Path(__file__).resolve()
    parent_root = script_path.parents[2]
    return parent_root / "Dolphin"


def summarize(times_ms):
    t = torch.tensor(times_ms, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std(unbiased=False)),
        "min_ms": float(t.min()),
        "max_ms": float(t.max()),
    }


def bench_gpu(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        with torch.no_grad():
            fn()

    torch.cuda.synchronize()
    times = []

    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            fn()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return summarize(times)


def bench_cpu(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        with torch.no_grad():
            fn()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return summarize(times)


def run_benchmark(model, device, warmup: int, repeat: int):
    model = model.to(device).eval()

    # 1-second audio at 16 kHz
    audio = torch.randn(1, 16000, device=device)

    # 1-second visual input: 25 FPS, grayscale, 88x88
    # Shape follows Dolphin Inference.py:
    # torch.from_numpy(window_mouth_roi[None, None])
    # => [B, C, T, H, W]
    video = torch.randn(1, 1, 25, 88, 88, device=device)

    def forward_once():
        return model(audio, video)

    if device.type == "cuda":
        return bench_gpu(forward_once, warmup, repeat)

    return bench_cpu(forward_once, warmup, repeat)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Dolphin full-model 1-second forward latency."
    )
    parser.add_argument(
        "--dolphin-root",
        default=None,
        help=(
            "Path to original Dolphin repo root. "
            "Default: sibling repo ../Dolphin/Dolphin relative to Dolphin-CHiME-9."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--cpu-repeat", type=int, default=20)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Set torch CPU threads for CPU benchmark.",
    )
    parser.add_argument(
        "--model-source",
        default="JusperLee/Dolphin",
        help=(
            "HF repo id or local directory containing config.json and model.safetensors. "
            "Default: JusperLee/Dolphin"
        ),
    )
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    dolphin_root = (
        Path(args.dolphin_root).expanduser().resolve()
        if args.dolphin_root is not None
        else default_dolphin_root()
    )

    if not (dolphin_root / "look2hear").exists():
        raise FileNotFoundError(
            f"Cannot find Dolphin repo at: {dolphin_root}\n"
            "Expected it to contain look2hear/.\n"
            "Use --dolphin-root /path/to/Dolphin/Dolphin if your layout differs."
        )

    sys.path.insert(0, str(dolphin_root))
    os.chdir(dolphin_root)

    from look2hear.models.dolphin import Dolphin

    torch.set_grad_enabled(False)

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, min(args.num_threads, 4)))

    print("torch num threads:", torch.get_num_threads())
    print("torch interop threads:", torch.get_num_interop_threads())

    print("=" * 80)
    print("Benchmark 1: Dolphin full-model forward latency")
    print("=" * 80)
    print(f"Dolphin root: {dolphin_root}")
    print("Input audio:  [1, 16000] = 1 second at 16 kHz")
    print("Input visual: [1, 1, 25, 88, 88] = 1 second at 25 FPS")
    print("Batch size:   1")
    print("Warmup:", args.warmup)
    print("Model source:", args.model_source)
    print("GPU repeat:", args.repeat)
    print("CPU repeat:", args.cpu_repeat)
    print("=" * 80)

    if not args.cpu_only and torch.cuda.is_available():
        model_gpu = Dolphin.from_pretrained(args.model_source).eval()
        gpu_result = run_benchmark(
            model_gpu,
            torch.device("cuda"),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print("\nGPU latency:")
        print(gpu_result)

    if not args.gpu_only:
        model_cpu = Dolphin.from_pretrained(args.model_source).eval()
        cpu_result = run_benchmark(
            model_cpu,
            torch.device("cpu"),
            warmup=min(args.warmup, 5),
            repeat=args.cpu_repeat,
        )
        print("\nCPU latency:")
        print(cpu_result)

    print("\nPaper Table 4 Dolphin reported latency:")
    print("GPU latency: 33.24 ms")
    print("CPU latency: 2117.96 ms")
    print("=" * 80)


if __name__ == "__main__":
    main()