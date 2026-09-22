import torch

# VGG19 feature layers commonly used for perceptual NST.
CONTENT_LAYER = "conv4_2"
STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")

LAYER_INDEX = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,
    "conv5_1": 28,
}


def extract_features(model, x: torch.Tensor, wanted_layers: set[str]):
    features = {}
    for index, layer in enumerate(model):
        x = layer(x)
        for name, layer_index in LAYER_INDEX.items():
            if layer_index == index and name in wanted_layers:
                features[name] = x
    return features


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    features = x.view(b, c, h * w)
    gram = features @ features.transpose(1, 2)
    return gram / (c * h * w)
