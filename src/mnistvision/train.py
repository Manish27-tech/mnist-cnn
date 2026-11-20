"""Training."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from mnistvision import config
from mnistvision.model import build_model, save_checkpoint


@dataclass
class TrainingResult:
    model: nn.Module
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    epochs_run: int = 0
    seconds: float = 0.0

    @property
    def summary(self):
        return {"best_epoch": self.best_epoch, "best_val_accuracy": round(self.best_val_accuracy, 4),
                "epochs_run": self.epochs_run, "seconds": round(self.seconds, 2),
                "seconds_per_epoch": round(self.seconds / max(self.epochs_run, 1), 2)}


def set_seed(seed=config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(model, train_loader, val_loader, *, epochs=config.EPOCHS, checkpoint_path=None, metadata=None, verbose=True):
    set_seed()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_state, best_acc, best_epoch = None, 0.0, 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            correct += (logits.argmax(1) == targets).sum().item()
            total += len(images)
        scheduler.step()
        model.eval()
        vloss = vcorrect = vtotal = 0
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images)
                vloss += criterion(logits, targets).item() * len(images)
                vcorrect += (logits.argmax(1) == targets).sum().item()
                vtotal += len(images)
        row = {"epoch": epoch, "train_loss": round(loss_sum / total, 4),
               "train_accuracy": round(correct / total, 4), "val_loss": round(vloss / vtotal, 4),
               "val_accuracy": round(vcorrect / vtotal, 4), "seconds": round(time.perf_counter() - started, 2)}
        history.append(row)
        if verbose:
            print(f"epoch {epoch}/{epochs}  train {row['train_loss']:.4f}/{row['train_accuracy']:.4f}  val {row['val_loss']:.4f}/{row['val_accuracy']:.4f}", flush=True)
        if row["val_accuracy"] > best_acc:
            best_acc = row["val_accuracy"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    result = TrainingResult(model, history, best_epoch, best_acc, len(history), time.perf_counter() - started)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, {
            "history": history, "best_epoch": best_epoch, "best_val_accuracy": best_acc,
            "epochs_run": len(history), "training_seconds": result.seconds,
            "train_rows": len(train_loader.dataset), "val_rows": len(val_loader.dataset),
            "batch_size": config.BATCH_SIZE, "seed": config.SEED, **(metadata or {}),
        })
    return result


def train_reference(dataset, *, epochs=config.EPOCHS, checkpoint_path=None, verbose=True):
    from mnistvision.data import build_loaders

    torch.set_num_threads(config.NUM_THREADS)
    train_loader, val_loader, test_loader = build_loaders(dataset)
    model = build_model("cnn")
    result = fit(model, train_loader, val_loader, epochs=epochs, checkpoint_path=checkpoint_path, verbose=verbose)
    return result, (train_loader, val_loader, test_loader)
