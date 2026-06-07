"""
training_engine.py — Real MobileNetV3-Small fine-tuning pipeline.

Phase 1: Freeze feature layers, train classifier head only.
Phase 2: Unfreeze last feature block, fine-tune with lower LR.

Streams per-epoch updates via an asyncio.Queue for WebSocket delivery.
"""

import asyncio
import time
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from database import (
    create_training_run, log_epoch, finish_training_run,
    save_model_version, set_active_model
)
from green_tracker import GreenMetricsTracker

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CLASSES = ["clothing", "ewaste", "glass", "metal", "organic", "paper", "plastic", "recyclable"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    # Freeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace the final classifier layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def unfreeze_last_block(model: nn.Module):
    """Unfreeze the last feature block for Phase 2 fine-tuning."""
    for param in model.features[-1].parameters():
        param.requires_grad = True


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, training=True):
    """Run one full epoch. Returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if training and optimizer:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training and optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ── Training session state (global singleton) ──────────────────────────────

class TrainingState:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.current_run_id: Optional[int] = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.history: list = []
        self.best_val_acc: float = 0.0
        self.active_model: Optional[nn.Module] = None

    def reset(self):
        self.is_running = False
        self.should_stop = False
        self.current_run_id = None
        self.history = []
        self.best_val_acc = 0.0


TRAINING_STATE = TrainingState()


# ── Main training function (runs in a thread) ──────────────────────────────

def train(
    train_loader,
    val_loader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    fine_tune: bool = True,
    fine_tune_epochs: int = 5,
    batch_size: int = 32,
    loop: asyncio.AbstractEventLoop = None,
):
    """
    Full training pipeline. Designed to run in a background thread.
    Posts progress updates to TRAINING_STATE.queue.
    """
    if TRAINING_STATE.is_running:
        return
    if train_loader is None:
        _post(loop, {"type": "error", "message": "No training data found. Add images to datasets/waste_dataset/train/"})
        return

    TRAINING_STATE.is_running = True
    TRAINING_STATE.should_stop = False
    TRAINING_STATE.history = []
    TRAINING_STATE.best_val_acc = 0.0

    # Create DB run
    run_id = create_training_run("centralized", notes=f"epochs={epochs}, lr={learning_rate}")
    TRAINING_STATE.current_run_id = run_id

    model = build_model(num_classes=len(CLASSES), pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_state = None
    patience_counter = 0
    patience = 5
    total_co2 = 0.0
    tracker = GreenMetricsTracker()

    _post(loop, {
        "type": "start",
        "run_id": run_id,
        "total_epochs": epochs + (fine_tune_epochs if fine_tune else 0),
        "device": str(DEVICE),
    })

    # ── Phase 1: Classifier head training ─────────────────────────────────
    all_epochs = epochs + (fine_tune_epochs if fine_tune else 0)
    phase_label = "Head Training"

    for epoch in range(1, epochs + 1):
        if TRAINING_STATE.should_stop:
            break

        t0 = time.time()
        tracker.start_tracking()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_loss, val_acc = (0.0, 0.0) if val_loader is None else run_epoch(
            model, val_loader, criterion, training=False
        )
        scheduler.step()

        green = tracker.stop_tracking()
        total_co2 += green.get("estimated_co2_kg", 0.0)
        duration = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # Early stopping
        if val_acc > TRAINING_STATE.best_val_acc:
            TRAINING_STATE.best_val_acc = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Log to DB
        log_epoch(run_id, epoch, train_loss, val_loss, train_acc, val_acc, lr_now, duration)

        entry = {
            "type": "epoch",
            "epoch": epoch,
            "total_epochs": all_epochs,
            "phase": phase_label,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "val_acc": round(val_acc * 100, 2),
            "learning_rate": round(lr_now, 6),
            "best_val_acc": round(TRAINING_STATE.best_val_acc * 100, 2),
            "co2_kg": round(total_co2, 6),
            "duration_s": round(duration, 2),
        }
        TRAINING_STATE.history.append(entry)
        _post(loop, entry)

        if patience_counter >= patience:
            _post(loop, {"type": "info", "message": f"Early stopping at epoch {epoch}"})
            break

    # ── Phase 2: Fine-tuning ───────────────────────────────────────────────
    if fine_tune and not TRAINING_STATE.should_stop:
        unfreeze_last_block(model)
        ft_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate / 10, weight_decay=1e-4
        )
        ft_scheduler = CosineAnnealingLR(ft_optimizer, T_max=fine_tune_epochs)
        phase_label = "Fine-Tuning"
        patience_counter = 0

        for ft_epoch in range(1, fine_tune_epochs + 1):
            if TRAINING_STATE.should_stop:
                break

            global_epoch = epochs + ft_epoch
            t0 = time.time()
            tracker.start_tracking()

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, ft_optimizer, training=True)
            val_loss, val_acc = (0.0, 0.0) if val_loader is None else run_epoch(
                model, val_loader, criterion, training=False)
            ft_scheduler.step()

            green = tracker.stop_tracking()
            total_co2 += green.get("estimated_co2_kg", 0.0)
            duration = time.time() - t0
            lr_now = ft_scheduler.get_last_lr()[0]

            if val_acc > TRAINING_STATE.best_val_acc:
                TRAINING_STATE.best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            log_epoch(run_id, global_epoch, train_loss, val_loss,
                      train_acc, val_acc, lr_now, duration)

            entry = {
                "type": "epoch",
                "epoch": global_epoch,
                "total_epochs": all_epochs,
                "phase": phase_label,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc * 100, 2),
                "val_acc": round(val_acc * 100, 2),
                "learning_rate": round(lr_now, 6),
                "best_val_acc": round(TRAINING_STATE.best_val_acc * 100, 2),
                "co2_kg": round(total_co2, 6),
                "duration_s": round(duration, 2),
            }
            TRAINING_STATE.history.append(entry)
            _post(loop, entry)

            if patience_counter >= patience:
                break

    # ── Save best model ────────────────────────────────────────────────────
    status = "stopped" if TRAINING_STATE.should_stop else "completed"
    total_trained = len(TRAINING_STATE.history)

    if best_model_state:
        if best_model_state:
            model.load_state_dict(best_model_state)
        version = _next_version()
        path = str(MODELS_DIR / f"model_{version}.pt")
        torch.save(model.state_dict(), path)
        save_model_version(
            version=version, path=path,
            train_acc=TRAINING_STATE.history[-1]["train_acc"] / 100 if TRAINING_STATE.history else 0.0,
            val_acc=TRAINING_STATE.best_val_acc,
            co2_kg=total_co2,
        )
        set_active_model(version)
        TRAINING_STATE.active_model = model
        _post(loop, {"type": "saved", "version": version, "path": path,
                     "val_acc": round(TRAINING_STATE.best_val_acc * 100, 2)})

    finish_training_run(run_id, TRAINING_STATE.best_val_acc, total_trained, status)
    _post(loop, {"type": "done", "status": status, "best_val_acc": round(
        TRAINING_STATE.best_val_acc * 100, 2), "total_co2_kg": round(total_co2, 6)})

    TRAINING_STATE.is_running = False


def _next_version() -> str:
    existing = list(MODELS_DIR.glob("model_v*.pt"))
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.replace("model_v", "")))
        except ValueError:
            pass
    return f"v{max(nums) + 1}" if nums else "v1"


def _post(loop, data: dict):
    """Thread-safe: put update onto the asyncio queue from a worker thread."""
    if loop and not loop.is_closed():
        asyncio.run_coroutine_threadsafe(TRAINING_STATE.queue.put(data), loop)
