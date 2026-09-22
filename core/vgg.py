import torch
from torchvision.models import VGG19_Weights, vgg19


def build_vgg19(device: torch.device):
    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
