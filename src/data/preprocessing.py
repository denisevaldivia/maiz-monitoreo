"""Preprocesamiento de una foto: recortes de 96 px -> tensor normalizado.

Es el mismo procedimiento que se uso para evaluar las fotos externas en el notebook
(ext_patches): lado corto a 256 px, malla de recortes y promedio de las predicciones.
"""

import io

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from src import config

eval_tf = T.Compose([T.Resize(config.IMG_SIZE), T.ToTensor(), T.Normalize(config.MEAN, config.STD)])


class InvalidImageError(ValueError):
    """La imagen recibida no se puede usar."""


def load_image(source):
    """Abre una imagen desde bytes o ruta y la valida. Devuelve PIL RGB."""
    try:
        if isinstance(source, bytes | bytearray):
            source = io.BytesIO(source)
        img = Image.open(source)
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("El archivo no es una imagen valida (usa JPEG o PNG).") from e
    if img.format not in config.ALLOWED_FORMATS:
        raise InvalidImageError(f"Formato {img.format} no permitido (usa JPEG o PNG).")
    if min(img.size) < config.MIN_SIDE:
        raise InvalidImageError(f"Imagen muy pequena: minimo {config.MIN_SIDE}px por lado.")
    return img.convert("RGB")


def extract_patches(img, k=config.EVAL_K, stride=config.PHOTO_STRIDE):
    """Reescala (lado corto = 256) y toma hasta k recortes repartidos en una malla."""
    patch = config.PATCH
    img = img.convert("RGB")
    w, h = img.size
    s = config.SOURCE_SIZE / min(w, h)
    new_size = (max(patch, round(w * s)), max(patch, round(h * s)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    W, H = img.size
    boxes = [
        (y, x) for y in range(0, H - patch + 1, stride) for x in range(0, W - patch + 1, stride)
    ]
    idx = np.unique(np.linspace(0, len(boxes) - 1, k).round().astype(int))
    return [img.crop((x, y, x + patch, y + patch)) for y, x in (boxes[i] for i in idx)]


def preprocess_image(img, k=config.EVAL_K):
    """Imagen PIL -> tensor (n_recortes, 3, IMG_SIZE, IMG_SIZE)."""
    return torch.stack([eval_tf(p) for p in extract_patches(img, k)])