import argparse
import csv
import time
from pathlib import Path

import torch

from core.gatys import GatysConfig, run_gatys
from core.image_utils import load_image, save_tensor
from core.vgg import build_vgg19


def main():
    parser = argparse.ArgumentParser(description="Run optimization-based neural style transfer.")
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
    vgg = build_vgg19(device)

    content = load_image(args.content, device, args.max_size)
    style = load_image(args.style, device, args.max_size)

    start = time.perf_counter()

    def progress(record):
        if record["step"] == 1 or record["step"] % 50 == 0 or record["step"] == args.steps:
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

    elapsed = time.perf_counter() - start

    save_tensor(generated, out / "stylized.png")

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with open(out / "run_metadata.txt", "w", encoding="utf-8") as f:
        f.write(f"device={device}\\n")
        f.write(f"steps={args.steps}\\n")
        f.write(f"max_size={args.max_size}\\n")
        f.write(f"elapsed_seconds={elapsed:.4f}\\n")
        if torch.cuda.is_available():
            f.write(f"gpu={torch.cuda.get_device_name(0)}\\n")

    print(f"Saved result to: {out / 'stylized.png'}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
