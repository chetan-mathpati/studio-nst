import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from core.johnson import (
    TransformerNet,
    VGGPerceptual,
    build_style_targets,
    compute_losses,
)


class FlatImageDataset(Dataset):
    def __init__(self, root, transform):
        self.files = [
            path
            for path in Path(root).rglob("*")
            if path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), 0


def image_loader(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )


def load_style(path, image_size, device):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )

    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--output", default="checkpoints/johnson")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=100000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = image_loader(args.image_size)

    if args.dataset.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
    else:
        dataset = FlatImageDataset(
            args.dataset,
            transform,
        )

    if len(dataset) == 0:
        raise SystemExit("No training images found.")

    if args.max_images and args.max_images < len(dataset):
        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            len(dataset),
            generator=generator,
        )[:args.max_images]

        dataset = Subset(
            dataset,
            indices.tolist(),
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    transformer = TransformerNet().to(device)
    perceptual = VGGPerceptual(device)

    optimizer = torch.optim.Adam(
        transformer.parameters(),
        lr=args.learning_rate,
    )

    style = load_style(
        args.style,
        args.image_size,
        device,
    )

    style_targets = build_style_targets(
        perceptual,
        style,
    )

    records = []
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        transformer.train()

        for content, _ in loader:
            content = content.to(
                device,
                non_blocking=True,
            )

            with torch.no_grad():
                content_features = perceptual(content)

            generated = transformer(content)

            content_loss, style_loss = compute_losses(
                perceptual,
                generated,
                content_features,
                style_targets,
            )

            total_loss = (
                args.content_weight * content_loss
                + args.style_weight * style_loss
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            total_loss.backward()
            optimizer.step()

            global_step += 1

            record = {
                "epoch": epoch + 1,
                "step": global_step,
                "total_loss": float(
                    total_loss.detach().cpu()
                ),
                "content_loss": float(
                    content_loss.detach().cpu()
                ),
                "style_loss": float(
                    style_loss.detach().cpu()
                ),
            }

            records.append(record)

            if (
                global_step == 1
                or global_step % 50 == 0
            ):
                print(
                    f"epoch={epoch + 1} "
                    f"step={global_step} "
                    f"total={record['total_loss']:.6f} "
                    f"content={record['content_loss']:.6f} "
                    f"style={record['style_loss']:.6e}"
                )

    elapsed = time.perf_counter() - start_time

    checkpoint = output_dir / "transformer_net.pth"

    torch.save(
        {
            "model_state_dict": transformer.state_dict(),
            "image_size": args.image_size,
            "residual_blocks": 5,
        },
        checkpoint,
    )

    metrics = output_dir / "training_metrics.csv"

    with open(
        metrics,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)

    metadata = output_dir / "training_metadata.txt"

    with open(
        metadata,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("device=" + str(device) + "\n")
        file.write("dataset=" + str(args.dataset) + "\n")
        file.write("image_size=" + str(args.image_size) + "\n")
        file.write("batch_size=" + str(args.batch_size) + "\n")
        file.write("epochs=" + str(args.epochs) + "\n")
        file.write("max_images=" + str(args.max_images) + "\n")
        file.write("content_weight=" + str(args.content_weight) + "\n")
        file.write("style_weight=" + str(args.style_weight) + "\n")
        file.write("learning_rate=" + str(args.learning_rate) + "\n")
        file.write("elapsed_seconds=" + str(round(elapsed, 4)) + "\n")

        if torch.cuda.is_available():
            file.write(
                "gpu=" + torch.cuda.get_device_name(0) + "\n"
            )

    print()
    print("Training completed.")
    print("Checkpoint: " + str(checkpoint))
    print("Metrics: " + str(metrics))
    print("Metadata: " + str(metadata))
    print("Elapsed: " + str(round(elapsed, 2)) + "s")
if __name__ == "__main__":
    main()
