"""
dataset_manager.py — Dataset scanning, stats, DataLoaders, and FL partitioning.
Expected folder structure:
  datasets/waste_dataset/{train,validation,test}/{class_name}/*.jpg
"""

import os
import base64
import io
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

DATASET_ROOT = Path("datasets/master_dataset")
CLASSES = ["clothing", "ewaste", "glass", "metal", "organic", "paper", "plastic", "recyclable"]
SPLITS = ["train", "validation", "test"]

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_images(directory: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.rglob("*") if f.suffix.lower() in exts
    )


def ensure_structure():
    """Create all dataset split/class folders if they don't exist."""
    for split in SPLITS:
        for cls in CLASSES:
            (DATASET_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    ensure_structure()
    stats = {"splits": {}, "total": 0, "classes": CLASSES}
    for split in SPLITS:
        split_data = {"total": 0, "per_class": {}}
        for cls in CLASSES:
            count = _count_images(DATASET_ROOT / split / cls)
            split_data["per_class"][cls] = count
            split_data["total"] += count
        stats["splits"][split] = split_data
        stats["total"] += split_data["total"]
    return stats


def get_thumbnails(split: str = "train", max_per_class: int = 6) -> dict:
    """Return base64-encoded thumbnail images per class."""
    thumbnails: dict[str, list[str]] = {cls: [] for cls in CLASSES}
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for cls in CLASSES:
        folder = DATASET_ROOT / split / cls
        if not folder.exists():
            continue
        files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        random.shuffle(files)
        for fpath in files[:max_per_class]:
            try:
                img = Image.open(fpath).convert("RGB")
                img.thumbnail((128, 128))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                thumbnails[cls].append(f"data:image/jpeg;base64,{b64}")
            except Exception:
                continue
    return thumbnails


# ── DataLoaders ───────────────────────────────────────────────────────────────

def _load_split(split: str, transform) -> Optional[datasets.ImageFolder]:
    folder = DATASET_ROOT / split
    if not folder.exists():
        return None
    try:
        ds = datasets.ImageFolder(str(folder), transform=transform)
        return ds if len(ds) > 0 else None
    except Exception:
        return None


def get_dataloaders(batch_size: int = 32, num_workers: int = 0):
    """Return (train_loader, val_loader, test_loader, class_names).
    Any missing/empty split returns None for that loader.
    """
    train_ds = _load_split("train", TRAIN_TRANSFORM)
    val_ds = _load_split("validation", EVAL_TRANSFORM)
    test_ds = _load_split("test", EVAL_TRANSFORM)

    def make_loader(ds, shuffle=False):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)

    class_names = train_ds.classes if train_ds else CLASSES
    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
        class_names,
    )


def partition_for_fl(num_nodes: int = 3, batch_size: int = 16, num_workers: int = 0):
    """Split the training set into `num_nodes` non-overlapping partitions.
    Returns list of DataLoaders, one per node.
    """
    train_ds = _load_split("train", TRAIN_TRANSFORM)
    if train_ds is None or len(train_ds) == 0:
        return [None] * num_nodes

    n = len(train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    chunk = n // num_nodes

    loaders = []
    for i in range(num_nodes):
        start = i * chunk
        end = start + chunk if i < num_nodes - 1 else n
        subset = Subset(train_ds, indices[start:end])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=False)
        loaders.append(loader)
    return loaders


# ── Synthetic dataset generator ───────────────────────────────────────────────

def generate_synthetic_dataset(images_per_class: int = 20):
    """Generate solid-color placeholder images so the training pipeline
    can be tested immediately even without a real dataset.
    """
    import numpy as np

    colors = {
        "Organic": (80, 140, 60),
        "Plastic": (60, 120, 200),
        "Paper": (220, 200, 160),
        "Mixed": (160, 100, 80),
        "Concealed_Polybag": (180, 180, 60),
    }
    split_ratios = {"train": 0.7, "validation": 0.2, "test": 0.1}

    ensure_structure()
    for cls, base_color in colors.items():
        all_imgs = []
        for i in range(images_per_class):
            # Add slight noise so images are not identical
            noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
            arr = np.clip(
                np.array(base_color, dtype=np.int16) + noise,
                0, 255
            ).astype(np.uint8)
            img = Image.fromarray(arr, "RGB")
            all_imgs.append(img)

        # Distribute across splits
        idx = 0
        for split, ratio in split_ratios.items():
            count = max(1, int(images_per_class * ratio))
            for j in range(count):
                if idx >= len(all_imgs):
                    break
                fpath = DATASET_ROOT / split / cls / f"synthetic_{idx:04d}.jpg"
                all_imgs[idx].save(str(fpath), "JPEG")
                idx += 1

    print(f"[DATASET] Synthetic dataset generated: {images_per_class} images/class")
    return get_stats()


if __name__ == "__main__":
    stats = generate_synthetic_dataset(30)
    print(stats)
