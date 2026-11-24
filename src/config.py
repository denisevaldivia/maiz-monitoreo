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
    return list(mapping["classes"])


CLASSES = load_classes()

# Preprocesamiento (igual que en el entrenamiento)
IMG_SIZE = 224  # tamano de entrada de la red
SOURCE_SIZE = 256  # tamano de las imagenes de PlantVillage
PATCH = 96  # lado del recorte
EVAL_K = 16  # recortes que se promedian por foto
STRIDE = 16  # paso de los recortes del indice de parches (entrenamiento)
PHOTO_STRIDE = 48  # paso de la malla de recortes en fotos nuevas
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Validacion de imagenes
ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_SIDE = 64
MAX_UPLOAD_MB = 10

# Servicio
MODEL_FILE = os.getenv("MODEL_FILE", "efficientnet_b0_v3.pt")
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.70"))
