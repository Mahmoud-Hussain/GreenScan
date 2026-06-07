"""
model_manager.py — Model versioning, loading, and activation management.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CLASSES = ["clothing", "ewaste", "glass", "metal", "organic", "paper", "plastic", "recyclable"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_base_model(num_classes: int = len(CLASSES)) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_model_version(path: str) -> nn.Module:
    """Load a saved model .pt file and return the model in eval mode."""
    model = _build_base_model(num_classes=len(CLASSES))
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def list_saved_models() -> list[dict]:
    """Scan the models/ directory and return metadata for all .pt files."""
    from database import get_all_model_versions
    return get_all_model_versions()


def get_active_model() -> tuple[nn.Module | None, str]:
    """
    Returns (model, version_string) of the currently active trained model,
    or (None, 'imagenet') if no trained model has been activated.
    """
    from database import get_active_model_path, get_all_model_versions
    path = get_active_model_path()
    if path and os.path.exists(path):
        try:
            model = load_model_version(path)
            versions = get_all_model_versions()
            active = next((v for v in versions if v["is_active"]), None)
            version_str = active["version"] if active else "trained"
            return model, version_str
        except Exception as e:
            print(f"[MODEL MANAGER] Could not load active model: {e}")
    return None, "imagenet"
