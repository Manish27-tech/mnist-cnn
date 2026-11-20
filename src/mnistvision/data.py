"""MNIST loading and loaders."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision import datasets, transforms

from mnistvision import config


def load_dataset():
    config.ensure_directories()
    cache = config.DATA / "mnist.npz"
    if cache.exists():
        with np.load(cache) as z:
            return Dataset(z["train_images"], z["train_labels"], z["test_images"], z["test_labels"])
    train = datasets.MNIST(root=str(config.DATA), train=True, download=True)
    test = datasets.MNIST(root=str(config.DATA), train=False, download=True)
    train_images = np.asarray(train.data.numpy(), dtype=np.uint8)
    train_labels = np.asarray(train.targets.numpy(), dtype=np.int64)
    test_images = np.asarray(test.data.numpy(), dtype=np.uint8)
    test_labels = np.asarray(test.targets.numpy(), dtype=np.int64)
    np.savez_compressed(cache, train_images=train_images, train_labels=train_labels,
                        test_images=test_images, test_labels=test_labels)
    return Dataset(train_images, train_labels, test_images, test_labels)


class Dataset:
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    @property
    def summary(self):
        return {"train_rows": int(len(self.train_labels)), "test_rows": int(len(self.test_labels))}


def stratified_split(labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in range(config.NUM_CLASSES):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        cut = int(round(len(idx) * (1 - val_fraction)))
        train_idx.extend(int(i) for i in idx[:cut])
        val_idx.extend(int(i) for i in idx[cut:])
    return np.asarray(train_idx), np.asarray(val_idx)


class _DS(TorchDataset):
    def __init__(self, images, labels, train: bool):
        self.images = images
        self.labels = labels
        self.train = train
        if train:
            self.tfm = transforms.Compose([
                transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize((config.PIXEL_MEAN,), (config.PIXEL_STD,)),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((config.PIXEL_MEAN,), (config.PIXEL_STD,)),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image

        return self.tfm(Image.fromarray(self.images[idx])), int(self.labels[idx])


def make_loader(images, labels, *, train, batch_size=config.BATCH_SIZE, seed=config.SEED):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(_DS(images, labels, train), batch_size=batch_size, shuffle=train,
                      num_workers=config.NUM_WORKERS, generator=gen)


def build_loaders(dataset, seed=config.SEED, batch_size=config.BATCH_SIZE):
    tr, va = stratified_split(dataset.train_labels, config.VAL_FRACTION, seed)
    return (
        make_loader(dataset.train_images[tr], dataset.train_labels[tr], train=True, batch_size=batch_size, seed=seed),
        make_loader(dataset.train_images[va], dataset.train_labels[va], train=False, batch_size=batch_size, seed=seed),
        make_loader(dataset.test_images, dataset.test_labels, train=False, batch_size=batch_size, seed=seed),
    )
