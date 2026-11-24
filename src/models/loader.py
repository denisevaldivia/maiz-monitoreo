"""Construccion del modelo y carga de pesos (local o desde S3)."""

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from src import config
from src.models.predictor import Predictor


def build_model(num_classes=None, pretrained=False):
    """EfficientNet-B0 con la cabeza cambiada a num_classes.

    OJO: debe ser la misma arquitectura que build('efficientnet_b0') del notebook.
    """
    num_classes = num_classes or len(config.CLASSES)
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    return net


def _download_from_s3(uri, dest):
    import boto3  # se importa aqui para no exigirlo en pruebas

    bucket, _, key = uri.removeprefix("s3://").partition("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))


def load_predictor(path=None, device="cpu"):
    """Carga los pesos de models/ (o los baja de MODEL_S3_URI si no estan)."""
    path = Path(path) if path else config.MODELS_DIR / config.MODEL_FILE
    if not path.exists():
        uri = os.getenv("MODEL_S3_URI")
        if not uri:
            raise FileNotFoundError(f"No hay modelo en {path} ni MODEL_S3_URI definido")
        _download_from_s3(uri, path)
    net = build_model()
    net.load_state_dict(torch.load(path, map_location=device))
    return Predictor(net, device=device)
