import torch
import torch.nn.functional as F


def calc_mean_std(features, eps=1e-5):
    if features.dim() != 4:
        raise ValueError("Expected a 4D tensor")

    mean = features.mean(dim=(2, 3), keepdim=True)
    variance = features.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = torch.sqrt(variance + eps)
    return mean, std


def adaptive_instance_normalization(content_features, style_features, eps=1e-5):
    if content_features.shape[0] != style_features.shape[0]:
        if style_features.shape[0] == 1:
            style_features = style_features.expand(content_features.shape[0], -1, -1, -1)
        else:
            raise ValueError("Content and style batch sizes must match or style batch size must be 1")

    if content_features.shape[1] != style_features.shape[1]:
        raise ValueError("Content and style channel counts must match")

    content_mean, content_std = calc_mean_std(content_features, eps)
    style_mean, style_std = calc_mean_std(style_features, eps)

    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean


def blend_features(content_features, transformed_features, alpha=1.0):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    return alpha * transformed_features + (1.0 - alpha) * content_features
