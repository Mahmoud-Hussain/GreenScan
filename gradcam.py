"""
gradcam.py — Gradient-weighted Class Activation Mapping for MobileNetV3-Small.
Returns a base64-encoded JPEG heatmap overlaid on the original image.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """
    Hooks into the last convolutional block of MobileNetV3-Small to
    compute Grad-CAM for any target class.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self._gradients: list = []
        self._activations: list = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        # MobileNetV3-Small: last feature block is model.features[-1]
        target_layer = self.model.features[-1]

        def fwd_hook(module, input, output):
            self._activations.append(output.detach())

        def bwd_hook(module, grad_in, grad_out):
            self._gradients.append(grad_out[0].detach())

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, image_bytes: bytes, class_idx: int = None) -> dict:
        """
        Run Grad-CAM on raw image bytes.
        Returns:
          - heatmap_b64: base64 JPEG of heatmap overlaid on original image
          - class_idx: the class that was explained
        """
        self._gradients.clear()
        self._activations.clear()

        # Preprocess
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original = pil_img.resize((224, 224))
        tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0)
        tensor.requires_grad_(False)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(probs.argmax().item())

        # Backward pass for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM computation
        if not self._gradients or not self._activations:
            return {"heatmap_b64": None, "class_idx": class_idx}

        grads = self._gradients[0]        # (1, C, H, W)
        acts = self._activations[0]       # (1, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize CAM to 224×224
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR
        )
        cam_arr = np.array(cam_img)

        # Apply colormap (jet: blue→red)
        heatmap = _apply_jet(cam_arr)

        # Blend with original
        orig_arr = np.array(original).astype(np.float32)
        blend = orig_arr * 0.5 + heatmap * 0.5
        blend = np.clip(blend, 0, 255).astype(np.uint8)
        blend_img = Image.fromarray(blend)

        buf = io.BytesIO()
        blend_img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "heatmap_b64": f"data:image/jpeg;base64,{b64}",
            "class_idx": class_idx,
        }


def _apply_jet(gray: np.ndarray) -> np.ndarray:
    """Convert a grayscale [0,255] array to a jet colormap RGB array."""
    t = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(t * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(t * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(t * 4 - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam(model: torch.nn.Module, image_bytes: bytes,
                     class_idx: int = None) -> dict:
    """Convenience wrapper — creates a GradCAM instance, runs it, removes hooks."""
    gc = GradCAM(model)
    try:
        result = gc.generate(image_bytes, class_idx=class_idx)
    finally:
        gc.remove_hooks()
    return result
