import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformerNet(nn.Module):
    def __init__(self, residual_blocks=5):
        super().__init__()
        self.input = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, 9),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(128) for _ in range(residual_blocks)]
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up2(x)
        return torch.tanh(self.output(x))


class VGGPerceptual(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        backbone = models.vgg19(weights=weights).features
        self.features = backbone[:29].to(device).eval()

        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.layer_map = {
            "conv1_1": 0,
            "conv2_1": 5,
            "conv3_1": 10,
            "conv4_1": 19,
            "conv4_2": 21,
            "conv5_1": 28,
        }

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.to(device)

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std

        outputs = {}

        for index, layer in enumerate(self.features):
            x = layer(x)

            for name, target in self.layer_map.items():
                if index == target:
                    outputs[name] = x

        return outputs


def gram_matrix(features):
    batch, channels, height, width = features.shape
    values = features.view(batch, channels, height * width)
    gram = torch.bmm(values, values.transpose(1, 2))
    return gram / (channels * height * width)


def build_style_targets(perceptual, style):
    with torch.no_grad():
        features = perceptual(style)

        return {
            name: gram_matrix(features[name]).detach()
            for name in [
                "conv1_1",
                "conv2_1",
                "conv3_1",
                "conv4_1",
                "conv5_1",
            ]
        }


def compute_losses(
    perceptual,
    generated,
    content_features,
    style_targets,
):
    features = perceptual(generated)

    content_loss = F.mse_loss(
        features["conv4_2"],
        content_features["conv4_2"],
    )

    style_loss = generated.new_zeros(())

    for name, target in style_targets.items():
        generated_gram = gram_matrix(features[name])
        target = target.expand_as(generated_gram)
        style_loss = style_loss + F.mse_loss(
            generated_gram,
            target,
        )

    return content_loss, style_loss
