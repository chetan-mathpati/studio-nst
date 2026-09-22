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
        description="Benchmark Gatys NST across image resolutions."
    )

    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", default="outputs/e003_resolution")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[256, 384, 512],
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vgg = build_vgg19(device)

    results = []

    for resolution in args.resolutions:
        print()
        print(f"Running resolution {resolution}px...")

        content = load_image(
            args.content,
            device,
            resolution,
        )

        style = load_image(
            args.style,
            device,
            resolution,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()

        def progress(record):
            if (
                record["step"] == 1
                or record["step"] % 100 == 0
                or record["step"] == args.steps
            ):
                print(
                    f"step={record['step']:4d} "
                    f"total={record['total_loss']:.4f} "
                    f"content={record['content_loss']:.4f} "
                    f"style={record['style_loss']:.6e}"
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
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

        resolution_dir = out / f"{resolution}px"
        resolution_dir.mkdir(parents=True, exist_ok=True)

        save_tensor(
            generated,
            resolution_dir / "stylized.png",
        )

        with open(
            resolution_dir / "metrics.csv",
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

        results.append(
            {
                "resolution": resolution,
                "steps": args.steps,
                "runtime_seconds": elapsed,
                "peak_gpu_memory_mb": peak_memory_mb,
                "final_total_loss": history[-1]["total_loss"],
                "final_content_loss": history[-1]["content_loss"],
                "final_style_loss": history[-1]["style_loss"],
            }
        )

        print(
            f"Completed {resolution}px | "
            f"runtime={elapsed:.4f}s | "
            f"content_loss={history[-1]['content_loss']:.6f} | "
            f"style_loss={history[-1]['style_loss']:.6e} | "
            f"peak_vram={peak_memory_mb:.2f} MB"
            if peak_memory_mb is not None
            else
            f"Completed {resolution}px | "
            f"runtime={elapsed:.4f}s | "
            f"content_loss={history[-1]['content_loss']:.6f} | "
            f"style_loss={history[-1]['style_loss']:.6e}"
        )

    results_file = out / "resolution_results.csv"

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

    metadata_file = out / "resolution_metadata.txt"

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"device={device}\n")
        f.write(f"steps={args.steps}\n")
        f.write(
            "resolutions="
            + ",".join(str(x) for x in args.resolutions)
            + "\n"
        )

        if torch.cuda.is_available():
            f.write(
                f"gpu={torch.cuda.get_device_name(0)}\n"
            )

    print()
    print("Resolution benchmark completed.")
    print(f"Results: {results_file}")
    print(f"Metadata: {metadata_file}")


if __name__ == "__main__":
    main()
