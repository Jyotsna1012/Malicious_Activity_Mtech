"""Compact MobileNetV3 + GRU model used only by the training/export tools."""
from __future__ import annotations


def build_model(pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    features = backbone.features
    del backbone

    class MobileNetGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.temporal = nn.GRU(input_size=576, hidden_size=128, batch_first=True)
            self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(128, 2))

        def forward(self, clips):
            batch, frames, channels, height, width = clips.shape
            images = clips.reshape(batch * frames, channels, height, width)
            features = self.pool(self.features(images)).flatten(1)
            features = features.reshape(batch, frames, -1)
            states, _ = self.temporal(features)
            return self.classifier(states[:, -1, :])

    return MobileNetGRU()
