"""The CNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MNISTCNN(nn.Module):
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.stem = ConvBlock(1, 32)
        self.block1 = ConvBlock(32, 64)
        self.block2 = ConvBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 10)
        self.gradcam_layer = self.block2.conv

    def forward(self, x):
        x = self.stem(x)
        x = F.max_pool2d(x, 2)
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def head(self):
        return self.fc


def build_model(name="cnn"):
    return MNISTCNN()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, metadata=None):
    payload = {"state_dict": model.state_dict(), "model_name": "cnn"}
    if metadata:
        payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(payload["model_name"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
