"""Preprocesamiento de una foto: recortes de 96 px sin fondo negro -> tensor normalizado.

Debe ser igual al de evaluacion del entrenamiento (PatchDataset con train=False).
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


def _dilate(mask):
    """Agranda la mascara 1 px en las 4 direcciones."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def extract_patches(img, k=config.EVAL_K):
    """Toma hasta k recortes fijos (repartidos) que casi no tengan pixeles negros."""
    size, patch = config.SOURCE_SIZE, config.PATCH
    img = img.convert("RGB").resize((size, size))
    black = _dilate(np.asarray(img).max(axis=2) < config.BLACK_THR)

    cands = []
    for y in range(0, size - patch + 1, config.STRIDE):
        for x in range(0, size - patch + 1, config.STRIDE):
            cands.append((black[y : y + patch, x : x + patch].mean(), y, x))

    valid = [c for c in cands if c[0] <= config.MAX_BLACK]
    if not valid:  # foto sin zona limpia: usa las menos oscuras
        valid = sorted(cands)[:k]

    idx = np.linspace(0, len(valid) - 1, num=min(k, len(valid))).round().astype(int)
    return [img.crop((x, y, x + patch, y + patch)) for _, y, x in (valid[i] for i in idx)]


def preprocess_image(img, k=config.EVAL_K):
    """Imagen PIL -> tensor (n_recortes, 3, IMG_SIZE, IMG_SIZE)."""
    return torch.stack([eval_tf(p) for p in extract_patches(img, k)])
