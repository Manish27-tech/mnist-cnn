"""Grad-CAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    heatmaps: np.ndarray


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.layer = model.gradcam_layer

    def generate(self, images, targets=None):
        self.model.eval()
        acts, grads = {}, {}

        def fwd(module, inp, out):
            acts["v"] = out

        def bwd(module, gin, gout):
            grads["v"] = gout[0]

        h1 = self.layer.register_forward_hook(fwd)
        h2 = self.layer.register_full_backward_hook(bwd)
        try:
            images = images.requires_grad_(True)
            logits = self.model(images)
            if targets is None:
                targets = logits.argmax(1)
            score = logits.gather(1, targets.view(-1, 1)).squeeze(1)
            self.model.zero_grad()
            score.backward(torch.ones_like(score))
            weights = grads["v"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts["v"]).sum(1))
            cam = F.interpolate(
                cam.unsqueeze(1), size=images.shape[2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            out = cam.detach().cpu().numpy()
            for i in range(len(out)):
                c = out[i]
                c -= c.min()
                if c.max() > 0:
                    c /= c.max()
                out[i] = c
        finally:
            h1.remove()
            h2.remove()
        return GradCAMResult(out)


def overlay(image, heatmap, alpha=0.45):
    import matplotlib.cm as cm

    colour = (cm.jet(np.clip(heatmap, 0, 1))[..., :3] * 255).astype(np.uint8)
    return ((1 - alpha) * image[..., None].astype(np.float32) + alpha * colour).astype(np.uint8)
