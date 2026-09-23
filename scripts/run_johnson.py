import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import torch
from PIL import Image
from torchvision import transforms

from core.johnson import TransformerNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--output", default="outputs/johnson/stylized.png")
    parser.add_argument("--max-size", type=int, default=512)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=False,
    )

    model = TransformerNet(
        residual_blocks=checkpoint.get("residual_blocks", 5)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.content).convert("RGB")

    width, height = image.size
    scale = min(1.0, args.max_size / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(tensor)

    output = (output.clamp(-1, 1) + 1.0) / 2.0
    output = output.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)

    print(f"Device: {device}")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
