from pathlib import Path

import torch
import torch.nn as nn


class AdaINVGG(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        weights_path = Path(weights_path)

        layers = [
            nn.Conv2d(3, 3, 1, 1, 0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3, 1, 0),
            nn.ReLU(inplace=True),
        ]

        self.features = nn.Sequential(*layers)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.features.load_state_dict(state)
        self.features = nn.Sequential(*list(self.features.children())[:31])

        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        return self.features(x)


class AdaINDecoder(nn.Module):
    def __init__(self, weights_path):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(state)

    def forward(self, x):
        return self.decoder(x)


class AdaINModel(nn.Module):
    def __init__(self, vgg_weights, decoder_weights, device):
        super().__init__()

        self.encoder = AdaINVGG(vgg_weights)
        self.decoder = AdaINDecoder(decoder_weights)

        self.encoder.to(device)
        self.decoder.to(device)

        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def stylize(self, content, style, alpha=1.0):
        content_features = self.encoder(content)
        style_features = self.encoder(style)

        content_mean = content_features.mean(dim=(2, 3), keepdim=True)
        content_std = content_features.var(
            dim=(2, 3), keepdim=True, unbiased=False
        ).add(1e-5).sqrt()

        style_mean = style_features.mean(dim=(2, 3), keepdim=True)
        style_std = style_features.var(
            dim=(2, 3), keepdim=True, unbiased=False
        ).add(1e-5).sqrt()

        normalized = (content_features - content_mean) / content_std
        transformed = normalized * style_std + style_mean

        blended = alpha * transformed + (1.0 - alpha) * content_features

        return self.decoder(blended)
