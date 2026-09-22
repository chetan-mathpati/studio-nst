import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import time

import torch

from core.gatys import GatysConfig, run_gatys
from core.image_utils import load_image, save_tensor
from core.vgg import build_vgg19


def run_experiment(vgg, content, style, steps, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    generated, history = run_gatys(
        vgg,
        content,
        style,
        GatysConfig(steps=steps),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_memory_mb = ""
    if torch.cuda.is_available():
        peak_memory_mb = round(
            torch.cuda.max_memory_allocated() / (1024 ** 2),
            2,
        )

    save_tensor(generated, output_dir / "stylized.png")

    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    final = history[-1]

    return {
        "steps": steps,
        "runtime_seconds": round(elapsed, 4),
        "peak_gpu_memory_mb": peak_memory_mb,
        "final_total_loss": final["total_loss"],
        "final_content_loss": final["content_loss"],
        "final_style_loss": final["style_loss"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Studio NST Gatys step-scaling benchmark."
    )
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--output", default="outputs/e002_benchmark")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[100, 300, 500],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vgg = build_vgg19(device)

    content = load_image(
        args.content,
        device,
        args.max_size,
    )

    style = load_image(
        args.style,
        device,
        args.max_size,
    )

    results = []

    for steps in args.steps:
        print()
        print(f"Running {steps} steps...")

        result = run_experiment(
            vgg,
            content,
            style,
            steps,
            output_root / f"{steps}_steps",
            device,
        )

        results.append(result)

        print(
            f"Completed {steps} steps | "
            f"runtime={result['runtime_seconds']}s | "
            f"content_loss={result['final_content_loss']:.6f} | "
            f"style_loss={result['final_style_loss']:.6e} | "
            f"peak_vram={result['peak_gpu_memory_mb']} MB"
        )

    results_file = output_root / "benchmark_results.csv"

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    metadata_file = output_root / "benchmark_metadata.txt"

    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"max_size={args.max_size}\n")
        f.write(f"content={args.content}\n")
        f.write(f"style={args.style}\n")

        if torch.cuda.is_available():
            f.write(f"gpu={torch.cuda.get_device_name(0)}\n")
            f.write(f"torch={torch.__version__}\n")
            f.write(f"cuda={torch.version.cuda}\n")

    print()
    print("Benchmark completed.")
    print(f"Results: {results_file}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
