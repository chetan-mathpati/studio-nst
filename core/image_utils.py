from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_image(path: str | Path, device: torch.device, max_size: int | None = 512) -> torch.Tensor:
    image = Image.open(path).convert("RGB")

    if max_size is not None:
        scale = min(1.0, max_size / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )

    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor.to(device)


def save_tensor(image: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = image.detach().cpu().clamp(0, 1).squeeze(0)
    transforms.ToPILImage()(image).save(path)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
