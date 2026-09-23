import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.adain_model import AdaINModel


def load_image(path, max_size):
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        size = (
            int(image.width * scale),
            int(image.height * scale),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)

    return transforms.ToTensor()(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AdaINModel(
        "weights/adain/vgg_normalised.pth",
        "weights/adain/decoder.pth",
        device,
    )

    content = load_image(args.content, args.max_size).to(device)
    style = load_image(args.style, args.max_size).to(device)

    start = time.perf_counter()

    output = model.stylize(
        content,
        style,
        alpha=args.alpha,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    output = output.clamp(0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(output.cpu(), str(output_path))

    print("Device: " + str(device))

    if torch.cuda.is_available():
        print("GPU: " + torch.cuda.get_device_name(0))
        print(
            "Peak GPU memory: "
            + str(round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2))
            + " MB"
        )

    print("Runtime: " + str(round(elapsed, 4)) + "s")
    print("Alpha: " + str(args.alpha))
    print("Saved result to: " + str(output_path))


if __name__ == "__main__":
    main()
