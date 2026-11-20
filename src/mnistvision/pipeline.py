"""End-to-end pipeline."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mnistvision import config
from mnistvision.data import build_loaders, load_dataset
from mnistvision.evaluate import (
    build_prediction_frame,
    calibration_summary,
    classification_summary,
    collect_logits,
    confidence_buckets,
    selective_accuracy,
    softmax_numpy,
    temperature_scale,
)
from mnistvision.explain import GradCAM, overlay
from mnistvision.model import count_parameters, load_checkpoint
from mnistvision.robustness import robustness_sweep
from mnistvision.train import train_reference


def _save(fig, name):
    path = config.FIGURES / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_json(name, payload):
    path = config.REPORTS / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _normalise(images):
    t = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    return (t - config.PIXEL_MEAN) / config.PIXEL_STD


def run_all(*, retrain=False, verbose=True):
    config.ensure_directories()
    dataset = load_dataset()

    # --- data figures ---
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(10), np.bincount(dataset.train_labels, minlength=10))
    ax.set_xticks(range(10))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_title("MNIST class balance")
    _save(fig, "01_class_distribution.png")

    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    for label, ax in enumerate(axes.flat):
        idx = int(np.flatnonzero(dataset.train_labels == label)[0])
        ax.imshow(dataset.train_images[idx], cmap="gray")
        ax.set_title(config.CLASS_NAMES[label], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "02_sample_grid.png")

    # --- train or reuse ---
    if retrain or not (config.CHECKPOINT.exists() and (config.REPORTS / "training_run.json").exists()):
        result, _ = train_reference(dataset, checkpoint_path=config.CHECKPOINT, verbose=verbose)
        training = {"history": result.history, "best_epoch": result.best_epoch,
                    "best_val_accuracy": result.best_val_accuracy, "epochs_run": result.epochs_run,
                    "seconds": result.seconds, "train_rows": 54000, "val_rows": 6000}
        _write_json("training_run.json", {"summary": result.summary, "history": result.history})
    else:
        _, meta = load_checkpoint(config.CHECKPOINT)
        training = {"history": meta["history"], "best_epoch": meta["best_epoch"],
                    "best_val_accuracy": meta["best_val_accuracy"], "epochs_run": meta["epochs_run"],
                    "seconds": meta["training_seconds"], "train_rows": meta.get("train_rows", 54000),
                    "val_rows": meta.get("val_rows", 6000)}
        if verbose:
            print("reusing checkpoint", flush=True)

    # --- evaluate ---
    model, _ = load_checkpoint(config.CHECKPOINT)
    _, val_loader, test_loader = build_loaders(dataset)
    val_logits, val_y = collect_logits(model, val_loader)
    test_logits, test_y = collect_logits(model, test_loader)
    temperature = temperature_scale(val_logits, val_y)
    calibration = calibration_summary(test_logits, test_y, temperature)
    classification = classification_summary(test_logits, test_y)
    calibrated = softmax_numpy(test_logits, temperature)
    preds = build_prediction_frame(calibrated, test_y)
    buckets = confidence_buckets(preds)
    selective = selective_accuracy(preds)

    # --- robustness ---
    picks = []
    for label in range(10):
        idx = np.flatnonzero(dataset.test_labels == label)
        picks.extend(int(i) for i in np.random.default_rng(config.SEED + label).choice(idx, size=100, replace=False))
    rob = robustness_sweep(model, dataset.test_images[np.asarray(picks)], dataset.test_labels[np.asarray(picks)])
    clean_acc = next(r["accuracy"] for r in rob if r["corruption"] == "clean")
    worst = min((r for r in rob if r["corruption"] != "clean"), key=lambda r: r["accuracy"])

    # --- gradcam ---
    chosen = pd.concat([preds.frame[preds.frame["correct"] == 1].nlargest(4, "confidence"),
                        preds.frame[preds.frame["correct"] == 0].nlargest(4, "confidence")], ignore_index=True)
    images = dataset.test_images[chosen["image_index"].to_numpy()]
    cam = GradCAM(model).generate(_normalise(images), torch.tensor(chosen["predicted_index"].to_numpy()))

    # --- figures ---
    hist = training["history"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train")
    axes[0].plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="val")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot([h["epoch"] for h in hist], [h["train_accuracy"] for h in hist], label="train")
    axes[1].plot([h["epoch"] for h in hist], [h["val_accuracy"] for h in hist], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "03_training_curves.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(classification["confusion_matrix"], cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "04_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(10)
    ax.bar(x - 0.2, [r["precision"] for r in classification["per_class"]], 0.2, label="precision")
    ax.bar(x, [r["recall"] for r in classification["per_class"]], 0.2, label="recall")
    ax.bar(x + 0.2, [r["f1"] for r in classification["per_class"]], 0.2, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.legend(frameon=False)
    _save(fig, "05_per_class_scores.png")

    before = calibration["bins_before"]
    after = calibration["bins"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for bins, label, color in ((before, "before", "#999"), (after, "after", "#4C72B0")):
        centres = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
        ax.plot(centres, [b["accuracy"] for b in bins], marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False)
    _save(fig, "06_reliability_diagram.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    df = pd.DataFrame(rob)
    for name, g in df[df["corruption"] != "clean"].groupby("corruption"):
        g = g.sort_values("severity")
        ax.plot(g["severity"], g["accuracy"], marker="o", label=name)
    ax.axhline(clean_acc, color="grey", linestyle="--", label=f"clean {clean_acc:.3f}")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("severity")
    ax.legend(frameon=False)
    _save(fig, "07_robustness.png")

    fig, axes = plt.subplots(2, 4, figsize=(9, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(overlay(images[i], cam.heatmaps[i]))
        ax.set_title(f"{chosen.iloc[i]['true_name']}->{chosen.iloc[i]['predicted_name']}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "08_gradcam.png")

    # --- feed ---
    kpis = {"accuracy": classification["accuracy"], "macro_f1": classification["macro_f1"],
            "parameters": count_parameters(model), "test_rows": len(test_y), "epochs_run": training["epochs_run"],
            "training_seconds": round(training["seconds"], 2), "temperature": temperature,
            "ece_before": calibration["ece_before"], "ece_after": calibration["ece_after"],
            "clean_accuracy": clean_acc, "worst_corruption": worst["corruption"],
            "worst_severity": float(worst["severity"]), "worst_accuracy": float(worst["accuracy"]),
            "worst_class": classification["worst_class"], "best_class": classification["best_class"],
            "top_confusion": classification["top_confusions"][0] if classification["top_confusions"] else None,
            "selective_90": next(r for r in selective if r["threshold"] == 0.9),
            "class_names": list(config.CLASS_NAMES), "seed": config.SEED,
            "dataset_source": config.DATASET_SOURCE, "architecture": "MNISTCNN (3 conv blocks)",
            "checkpoint_bytes": config.CHECKPOINT.stat().st_size}
    (config.DASHBOARD_FEED / "kpis.json").write_text(json.dumps(kpis, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(classification["per_class"]).to_csv(config.DASHBOARD_FEED / "per_class.csv", index=False)
    pd.DataFrame(hist).to_csv(config.DASHBOARD_FEED / "history.csv", index=False)
    pd.DataFrame(rob).to_csv(config.DASHBOARD_FEED / "robustness.csv", index=False)
    pd.DataFrame(selective).to_csv(config.DASHBOARD_FEED / "selective_accuracy.csv", index=False)
    pd.DataFrame(buckets).to_csv(config.DASHBOARD_FEED / "confidence_buckets.csv", index=False)
    preds.frame.to_csv(config.DASHBOARD_FEED / "predictions.csv", index=False)
    sample_idx = []
    for label in range(10):
        sample_idx.extend(int(i) for i in np.flatnonzero(dataset.test_labels == label)[:6])
    np.savez_compressed(config.DASHBOARD_FEED / "samples.npz", images=dataset.test_images[np.asarray(sample_idx)],
                        labels=dataset.test_labels[np.asarray(sample_idx)], indices=np.asarray(sample_idx))
    np.savez_compressed(config.DASHBOARD_FEED / "gradcam.npz", images=images, heatmaps=cam.heatmaps)

    _write_json("metrics.json", {"accuracy": classification["accuracy"], "macro_f1": classification["macro_f1"],
                                  "parameters": kpis["parameters"], "ece_before": calibration["ece_before"],
                                  "ece_after": calibration["ece_after"], "temperature": temperature,
                                  "worst_corruption": worst["corruption"], "worst_accuracy": worst["accuracy"]})
    _write_json("classification.json", classification)
    _write_json("calibration.json", calibration)
    _write_json("robustness.json", rob)

    (config.REPORTS / "REPORT.md").write_text(
        f"# MNIST CNN\n\n- Accuracy: {classification['accuracy']:.4f}\n- Macro F1: {classification['macro_f1']:.4f}\n"
        f"- ECE: {calibration['ece_before']:.4f} -> {calibration['ece_after']:.4f}\n"
        f"- Worst corruption: {worst['corruption']} at {worst['severity']} -> {worst['accuracy']:.4f}\n",
        encoding="utf-8")
    (config.REPORTS / "MODEL_CARD.md").write_text(
        f"# Model card\n\nA compact CNN for MNIST. Test accuracy {classification['accuracy']:.4f}. "
        "Intended for demonstration and teaching, not production use.\n", encoding="utf-8")
    return {"accuracy": classification["accuracy"], "figures": sorted(p.name for p in config.FIGURES.glob("*.png"))}
