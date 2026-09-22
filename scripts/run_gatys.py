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


def main():
    parser = argparse.ArgumentParser(
        description="Run optimization-based neural style transfer."
    )

    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--output", default="outputs/gatys")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    def progress(record):
        if (
            record["step"] == 1
            or record["step"] % 50 == 0
            or record["step"] == args.steps
        ):
            print(
                f"step={record['step']:4d} "
                f"total={record['total_loss']:.4f} "
                f"content={record['content_loss']:.4f} "
                f"style={record['style_loss']:.4f}"
            )

    generated, history = run_gatys(
        vgg,
        content,
        style,
        GatysConfig(steps=args.steps),
        progress_callback=progress,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_memory_mb = None

    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    output_image = out / "stylized.png"
    save_tensor(generated, output_image)

    metrics_file = out / "metrics.csv"

    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    metadata_file = out / "run_metadata.txt"

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"device={device}\n")
        f.write(f"steps={args.steps}\n")
        f.write(f"max_size={args.max_size}\n")
        f.write(f"elapsed_seconds={elapsed:.4f}\n")

        if torch.cuda.is_available():
            f.write(
                f"gpu={torch.cuda.get_device_name(0)}\n"
            )
            f.write(
                f"peak_gpu_memory_mb={peak_memory_mb:.2f}\n"
            )

    print()
    print("Experiment completed.")
    print(f"Saved result to: {output_image}")
    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved metadata to: {metadata_file}")
    print(f"Elapsed: {elapsed:.2f}s")

    if peak_memory_mb is not None:
        print(
            f"Peak GPU memory: {peak_memory_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()