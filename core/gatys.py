from dataclasses import dataclass

import torch

from .features import CONTENT_LAYER, STYLE_LAYERS, extract_features, gram_matrix
from .image_utils import normalize_batch


@dataclass
class GatysConfig:
    steps: int = 300
    content_weight: float = 1.0
    style_weight: float = 1e6
    learning_rate: float = 0.02


def run_gatys(vgg, content, style, config: GatysConfig, progress_callback=None):
    wanted = {CONTENT_LAYER, *STYLE_LAYERS}

    with torch.no_grad():
        content_features = extract_features(vgg, normalize_batch(content), wanted)
        style_features = extract_features(vgg, normalize_batch(style), wanted)
        style_targets = {k: gram_matrix(v) for k, v in style_features.items()}

    generated = content.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=config.learning_rate)

    history = []

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        generated_features = extract_features(vgg, normalize_batch(generated), wanted)

        content_loss = torch.nn.functional.mse_loss(
            generated_features[CONTENT_LAYER],
            content_features[CONTENT_LAYER],
        )

        style_loss = 0.0
        for layer in STYLE_LAYERS:
            style_loss = style_loss + torch.nn.functional.mse_loss(
                gram_matrix(generated_features[layer]),
                style_targets[layer],
            )

        total = config.content_weight * content_loss + config.style_weight * style_loss
        total.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0, 1)

        record = {
            "step": step,
            "total_loss": float(total.detach().cpu()),
            "content_loss": float(content_loss.detach().cpu()),
            "style_loss": float(style_loss.detach().cpu()),
        }
        history.append(record)

        if progress_callback:
            progress_callback(record)

    return generated.detach(), history
