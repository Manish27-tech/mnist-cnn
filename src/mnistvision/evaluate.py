"""Evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mnistvision import config


def collect_logits(model, loader):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for images, y in loader:
            logits.append(model(images))
            targets.append(y)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def softmax_numpy(logits, temperature=1.0):
    z = logits / max(temperature, 1e-6)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature_scale(val_logits, val_targets):
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate(([1.0], np.geomspace(0.5, 5.0, 400))):
        p = softmax_numpy(val_logits, t)
        nll = -np.log(p[np.arange(len(val_targets)), val_targets] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def expected_calibration_error(probs, targets, bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    ece = 0.0
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else 0.0
        c = float(conf[mask].mean()) if n else (lo + hi) / 2
        out.append({"bin_lower": lo, "bin_upper": hi, "count": n, "accuracy": acc, "confidence": c, "gap": abs(acc - c)})
        if n:
            ece += (n / len(targets)) * abs(acc - c)
    mce = max((abs(b["accuracy"] - b["confidence"]) for b in out if b["count"]), default=0.0)
    return ece, mce, out


def classification_summary(logits, targets):
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    preds = logits.argmax(1)
    p, r, f, s = precision_recall_fscore_support(targets, preds, labels=list(range(10)), zero_division=0)
    per_class = [{"class_index": i, "class_name": config.CLASS_NAMES[i], "precision": float(p[i]),
                  "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(10)]
    cm = confusion_matrix(targets, preds, labels=list(range(10))).tolist()
    top = [{"true_index": t, "true_name": config.CLASS_NAMES[t], "predicted_index": q,
            "predicted_name": config.CLASS_NAMES[q], "count": int(cm[t][q])}
           for t in range(10) for q in range(10) if t != q and cm[t][q] > 0]
    top.sort(key=lambda x: x["count"], reverse=True)
    worst = min(per_class, key=lambda x: x["f1"])
    best = max(per_class, key=lambda x: x["f1"])
    return {"accuracy": float((preds == targets).mean()),
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "top1_errors": int((preds != targets).sum()), "per_class": per_class,
            "confusion_matrix": cm, "top_confusions": top[:10],
            "worst_class": worst["class_name"], "best_class": best["class_name"],
            "label_order": list(config.CLASS_NAMES)}


@dataclass
class PredictionFrame:
    frame: object
    accuracy: float


def build_prediction_frame(probs, targets):
    import pandas as pd

    preds = probs.argmax(1)
    conf = probs.max(1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    frame = pd.DataFrame({"image_index": np.arange(len(targets)), "true_index": targets,
                          "true_name": [config.CLASS_NAMES[int(t)] for t in targets],
                          "predicted_index": preds,
                          "predicted_name": [config.CLASS_NAMES[int(p)] for p in preds],
                          "confidence": conf, "entropy": entropy,
                          "correct": (preds == targets).astype(int)})
    return PredictionFrame(frame, float((preds == targets).mean()))


def confidence_buckets(predictions, bins=10):
    rows = []
    frame = predictions.frame
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (frame["confidence"] >= lo) & (frame["confidence"] <= hi) if i == 0 else (frame["confidence"] > lo) & (frame["confidence"] <= hi)
        kept = frame[mask]
        rows.append({"lower": lo, "upper": hi, "count": len(kept),
                     "accuracy": float(kept["correct"].mean()) if len(kept) else 0.0})
    return rows


def selective_accuracy(predictions, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)):
    frame = predictions.frame
    return [{"threshold": t, "coverage": round(len(frame[frame["confidence"] >= t]) / len(frame), 4),
             "accuracy": round(float(frame[frame["confidence"] >= t]["correct"].mean()), 4) if (frame["confidence"] >= t).any() else 0.0,
             "count": int((frame["confidence"] >= t).sum())} for t in thresholds]


def calibration_summary(logits, targets, temperature):
    before = softmax_numpy(logits, 1.0)
    after = softmax_numpy(logits, temperature)
    ece_b, mce_b, bins_b = expected_calibration_error(before, targets)
    ece_a, mce_a, bins_a = expected_calibration_error(after, targets)
    return {"temperature": temperature, "accuracy": float((logits.argmax(1) == targets).mean()),
            "ece_before": ece_b, "ece_after": ece_a, "mce_before": mce_b, "mce_after": mce_a,
            "mean_confidence_before": float(before.max(1).mean()), "mean_confidence_after": float(after.max(1).mean()),
            "nll_before": float(-np.log(before[np.arange(len(targets)), targets] + 1e-12).mean()),
            "nll_after": float(-np.log(after[np.arange(len(targets)), targets] + 1e-12).mean()),
            "bins_before": bins_b, "bins": bins_a}
