"""Constantes compartidas por el entrenamiento, las pruebas y la API."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

# Orden por defecto (alfabetico). Si existe models/class_mapping.json se usa ese.
DEFAULT_CLASSES = ["common_rust", "gray_leaf_spot", "healthy", "northern_blight"]


def load_classes(mapping_file=None):
    """Lee el orden de clases del class_mapping.json usado en el entrenamiento."""
    f = Path(mapping_file) if mapping_file else MODELS_DIR / "class_mapping.json"
    if not f.exists():
        return list(DEFAULT_CLASSES)
    mapping = json.loads(f.read_text())
    if all(isinstance(v, int) for v in mapping.values()):  # {"healthy": 0, ...}
        return [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    return [v for _, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))]  # {"0": "healthy"}


CLASSES = load_classes()

# Preprocesamiento (igual que en el entrenamiento)
IMG_SIZE = 224  # tamano de entrada de la red
SOURCE_SIZE = 256  # tamano de las imagenes de PlantVillage
PATCH = 96  # lado del recorte
EVAL_K = 8  # recortes que se promedian por imagen
STRIDE = 16  # paso al buscar recortes
MAX_BLACK = 0.005  # fraccion maxima de pixeles negros por recorte
BLACK_THR = 10  # un pixel con max(R,G,B) menor a esto cuenta como negro
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Validacion de imagenes
ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_SIDE = 64
MAX_UPLOAD_MB = 10

# Servicio
MODEL_FILE = os.getenv("MODEL_FILE", "efficientnet_b0_v3.pt")
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.70"))
