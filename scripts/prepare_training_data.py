import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="assets/content")
    parser.add_argument("--output", default="assets/training")
    parser.add_argument("--copies", type=int, default=8)

    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    images = [
        path for path in source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    if not images:
        raise SystemExit("No source images found.")

    created = 0

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        for index in range(args.copies):
            target = output / f"{image_path.stem}_{index:03d}.jpg"
            image.save(target, quality=95)
            created += 1

    print(f"Created {created} training images in {output}")


if __name__ == "__main__":
    main()
