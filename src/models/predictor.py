"""Prediccion sobre una imagen: promedia las probabilidades de varios recortes."""

import torch

from src import config
from src.data.preprocessing import preprocess_image


class Predictor:
    def __init__(self, model, classes=None, device="cpu", low_confidence=None):
        self.model = model.to(device).eval()
        self.classes = classes or config.CLASSES
        self.device = device
        self.low_confidence = low_confidence or config.LOW_CONFIDENCE

    @torch.no_grad()
    def predict(self, img):
        x = preprocess_image(img).to(self.device)
        probs = torch.softmax(self.model(x), dim=1).mean(dim=0).cpu()
        conf, idx = probs.max(dim=0)
        named = zip(self.classes, probs, strict=True)
        return {
            "predicted_class": self.classes[int(idx)],
            "confidence": round(float(conf), 4),
            "probabilities": {c: round(float(p), 4) for c, p in named},
            "needs_review": bool(conf < self.low_confidence),
            "n_patches": int(x.shape[0]),
        }
