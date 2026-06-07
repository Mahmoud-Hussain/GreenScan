"""
waste_model.py — Inference engine.
Loads the fine-tuned model if one is active; falls back to ImageNet keyword
mapping if no trained model exists yet.
"""

import io
import json
import urllib.request

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

WASTE_CLASSES = ["Clothing", "E-Waste", "Glass", "Metal", "Organic", "Paper", "Plastic", "Recyclable"]
# Internal class names used by ImageFolder (underscores, no spaces) — must match folder order
FOLDER_CLASSES = ["clothing", "ewaste", "glass", "metal", "organic", "paper", "plastic", "recyclable"]

CATEGORY_DESCRIPTIONS = {
    "Clothing": "Textiles and fabrics. Can often be donated or recycled at specialized facilities.",
    "E-Waste": "Electronic waste. Contains valuable and hazardous materials. Must be taken to an e-waste recycler.",
    "Glass": "Glass items. Highly recyclable but should be separated by color depending on local rules.",
    "Metal": "Metal cans and objects. Highly recyclable.",
    "Organic": "Biodegradable material such as food scraps and garden waste. Can be composted.",
    "Paper": "Cellulose-based material. Recyclable when clean and dry.",
    "Plastic": "Synthetic polymer-based material (bottles, containers, packaging). Check the recycling number before disposal.",
    "Recyclable": "General recyclable material (Kaggle Dataset). Items such as paper, plastic bottles, metal cans.",
    "clothing": "Textiles and fabrics. Can often be donated or recycled at specialized facilities.",
    "ewaste": "Electronic waste. Contains valuable and hazardous materials. Must be taken to an e-waste recycler.",
    "glass": "Glass items. Highly recyclable but should be separated by color depending on local rules.",
    "metal": "Metal cans and objects. Highly recyclable.",
    "organic": "Biodegradable material such as food scraps and garden waste. Can be composted.",
    "paper": "Cellulose-based material. Recyclable when clean and dry.",
    "plastic": "Synthetic polymer-based material (bottles, containers, packaging). Check the recycling number before disposal.",
    "recyclable": "General recyclable material (Kaggle Dataset). Items such as paper, plastic bottles, metal cans.",
}

KEYWORD_MAP = {
    "banana": "Organic", "apple": "Organic", "orange": "Organic",
    "lemon": "Organic", "strawberry": "Organic", "mushroom": "Organic",
    "broccoli": "Organic", "cauliflower": "Organic", "cucumber": "Organic",
    "eggplant": "Organic", "artichoke": "Organic", "fig": "Organic",
    "pineapple": "Organic", "pomegranate": "Organic", "corn": "Organic",
    "acorn": "Organic", "leaf": "Organic", "grass": "Organic",
    "hay": "Organic", "wood": "Organic", "flower": "Organic",
    "vegetable": "Organic", "fruit": "Organic", "bread": "Organic",
    "pretzel": "Organic", "bagel": "Organic", "pizza": "Organic",
    "hotdog": "Organic", "egg": "Organic",
    "bottle": "Plastic", "water bottle": "Plastic", "plastic": "Plastic",
    "pop bottle": "Plastic", "syringe": "Plastic", "bucket": "Plastic",
    "cup": "Plastic", "jerrycan": "Plastic", "jug": "Plastic",
    "beaker": "Plastic", "pill bottle": "Plastic", "rubber": "Plastic",
    "hose": "Plastic", "toothbrush": "Plastic", "comb": "Plastic",
    "envelope": "Paper", "book jacket": "Paper", "comic book": "Paper",
    "newspaper": "Paper", "notebook": "Paper", "paper towel": "Paper",
    "cardboard": "Paper", "box": "Paper", "book": "Paper", "magazine": "Paper",
    "shopping bag": "Concealed-Polybag", "garbage bag": "Concealed-Polybag",
    "plastic bag": "Concealed-Polybag", "sack": "Concealed-Polybag",
    "backpack": "Concealed-Polybag", "bag": "Concealed-Polybag",
    "purse": "Concealed-Polybag", "mailbag": "Concealed-Polybag",
    "garbage truck": "Mixed Waste", "trash can": "Mixed Waste",
    "ashcan": "Mixed Waste", "wastebasket": "Mixed Waste",
    "can": "Mixed Waste", "tin": "Mixed Waste", "battery": "Mixed Waste",
    "keyboard": "Mixed Waste", "television": "Mixed Waste",
}
CATEGORY_FALLBACK = "Mixed Waste"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WasteClassifier:
    """
    Unified classifier:
    - If a fine-tuned model is active → uses it (5-class softmax)
    - Otherwise → uses ImageNet MobileNetV3 + keyword voting
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._imagenet_loaded = False
            cls._instance._trained_model = None
            cls._instance._trained_version = None
        return cls._instance

    # ── ImageNet fallback ──────────────────────────────────────────────────
    def _load_imagenet(self):
        if self._imagenet_loaded:
            return
        print("[MODEL] Loading MobileNetV3-Small (ImageNet)...")
        self._imagenet_model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self._imagenet_model.eval()
        url = (
            "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
            "/master/imagenet-simple-labels.json"
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                self.imagenet_labels = json.loads(r.read().decode())
        except Exception:
            self.imagenet_labels = [str(i) for i in range(1000)]
        self._imagenet_loaded = True
        print("[MODEL] ImageNet model ready.")

    def _map_label(self, label: str) -> str:
        label_lower = label.lower()
        for keyword, cat in KEYWORD_MAP.items():
            if keyword in label_lower:
                return cat
        return CATEGORY_FALLBACK

    # ── Trained model loading ──────────────────────────────────────────────
    def refresh_trained(self):
        """Called after training completes or on predict to load latest active model."""
        from model_manager import get_active_model
        model, version = get_active_model()
        if model is not None:
            self._trained_model = model
            self._trained_version = version

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, image_bytes: bytes) -> dict:
        self.refresh_trained()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = EVAL_TRANSFORM(img).unsqueeze(0)

        if self._trained_model is not None:
            return self._predict_trained(tensor, image_bytes)
        else:
            self._load_imagenet()
            return self._predict_imagenet(tensor)

    def _predict_trained(self, tensor: torch.Tensor, image_bytes: bytes) -> dict:
        model = self._trained_model
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = torch.topk(probs, min(5, len(FOLDER_CLASSES)))
        top5 = [
            {"class": FOLDER_CLASSES[i], "confidence": round(p, 4)}
            for i, p in zip(top5_idx.tolist(), top5_probs.tolist())
        ]
        best_idx = int(probs.argmax().item())
        folder_cls = FOLDER_CLASSES[best_idx]
        # Map folder name → display name
        display_cls = folder_cls.replace("_", " ").replace("Concealed Polybag", "Concealed-Polybag")
        if folder_cls == "Mixed":
            display_cls = "Mixed Waste"

        # Grad-CAM
        gradcam_b64 = None
        try:
            from gradcam import generate_gradcam
            gc = generate_gradcam(model, image_bytes, class_idx=best_idx)
            gradcam_b64 = gc.get("heatmap_b64")
        except Exception:
            pass

        return {
            "predicted_class": display_cls,
            "confidence": round(float(probs[best_idx].item()), 4),
            "category_scores": {
                FOLDER_CLASSES[i].replace("_", " "): round(float(probs[i].item()), 4)
                for i in range(len(FOLDER_CLASSES))
            },
            "top5": top5,
            "description": CATEGORY_DESCRIPTIONS.get(folder_cls, ""),
            "model_version": self._trained_version or "trained",
            "gradcam_b64": gradcam_b64,
        }

    def _predict_imagenet(self, tensor: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self._imagenet_model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, 10)
        category_scores: dict[str, float] = {c: 0.0 for c in WASTE_CLASSES}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            label = self.imagenet_labels[idx]
            cat = self._map_label(label)
            category_scores[cat] += prob

        best = max(category_scores, key=category_scores.get)
        total = sum(category_scores.values()) or 1.0
        conf = round(min(category_scores[best] / total, 0.99), 4)

        top5_imagenet = [
            {"class": self.imagenet_labels[i], "confidence": round(p, 4)}
            for p, i in zip(top_probs.tolist()[:5], top_indices.tolist()[:5])
        ]
        return {
            "predicted_class": best,
            "confidence": conf,
            "category_scores": {k: round(v / total, 4) for k, v in category_scores.items()},
            "top5": top5_imagenet,
            "description": CATEGORY_DESCRIPTIONS.get(best, ""),
            "model_version": "imagenet",
            "gradcam_b64": None,
        }


classifier = WasteClassifier()
