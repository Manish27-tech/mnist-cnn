"""Robustness sweep."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from mnistvision import config


def apply_corruption(images, corruption, severity, seed=0):
    rng = np.random.default_rng(seed)
    x = images.astype(np.float32) / 255.0
    if corruption == "noise":
        x = x + rng.normal(0, severity, x.shape).astype(np.float32)
    elif corruption == "blur":
        x = np.stack([gaussian_filter(im, sigma=severity) for im in x])
    elif corruption == "contrast":
        mean = x.mean(axis=(1, 2), keepdims=True)
        x = mean + severity * (x - mean)
    elif corruption == "occlusion":
        size = int(severity)
        h, w = x.shape[1:]
        top, left = (h - size) // 2, (w - size) // 2
        x[:, top:top + size, left:left + size] = 0.0
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _acc(model, tensor, labels):
    with torch.no_grad():
        preds = model(tensor).argmax(1).numpy()
    return float((preds == labels).mean())


def robustness_sweep(model, images, labels, seed=config.SEED):
    def tensor(imgs):
        t = torch.from_numpy(imgs).float().unsqueeze(1) / 255.0
        return (t - config.PIXEL_MEAN) / config.PIXEL_STD

    clean = _acc(model, tensor(images), labels)
    rows = [{"corruption": "clean", "severity": 0.0, "accuracy": clean, "delta": 0.0}]
    for name, sevs in {"noise": [0.1, 0.2, 0.3], "blur": [0.5, 1.0, 1.5],
                       "contrast": [0.5, 0.75, 1.25], "occlusion": [8, 12, 16]}.items():
        for sev in sevs:
            acc = _acc(model, tensor(apply_corruption(images, name, sev, seed)), labels)
            rows.append({"corruption": name, "severity": float(sev), "accuracy": acc, "delta": acc - clean})
    return rows
