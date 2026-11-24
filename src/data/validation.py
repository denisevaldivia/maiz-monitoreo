"""Validacion de datos: imagenes, etiquetas y fuga entre train/val/test.

Cada funcion devuelve una lista de problemas (vacia = todo bien).

Uso desde la terminal:
    python -m src.data.validation --csv data/split_v1.csv --root data/raw
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src import config


def validate_image_file(path):
    """Revisa que el archivo abra, sea JPEG/PNG y no sea demasiado chico."""
    path = Path(path)
    if not path.exists():
        return [f"{path}: no existe"]
    try:
        with Image.open(path) as img:
            img.load()
            fmt, size = img.format, img.size
    except (UnidentifiedImageError, OSError):
        return [f"{path}: no se puede abrir como imagen"]
    problems = []
    if fmt not in config.ALLOWED_FORMATS:
        problems.append(f"{path}: formato {fmt} no permitido")
    if min(size) < config.MIN_SIDE:
        problems.append(f"{path}: tamano {size} menor a {config.MIN_SIDE}px")
    return problems


def validate_images(paths):
    problems = []
    for p in paths:
        problems += validate_image_file(p)
    return problems


def validate_labels(df, classes=None, label_col="label"):
    """Todas las etiquetas deben estar en la lista de clases y no haber vacias."""
    classes = set(classes or config.CLASSES)
    problems = []
    nulls = int(df[label_col].isna().sum())
    if nulls:
        problems.append(f"{nulls} filas sin etiqueta")
    bad = sorted(set(df[label_col].dropna()) - classes)
    if bad:
        problems.append(f"etiquetas no validas: {bad}")
    return problems


def find_leakage(df, path_col="path", split_col="split", group_col="group"):
    """Ninguna imagen (ni grupo de duplicados) puede estar en dos conjuntos."""
    problems = []
    n_splits = df.groupby(path_col)[split_col].nunique()
    for p in n_splits[n_splits > 1].index:
        problems.append(f"imagen en mas de un conjunto: {p}")
    if group_col in df.columns:
        n_splits = df.groupby(group_col)[split_col].nunique()
        for g in n_splits[n_splits > 1].index:
            problems.append(f"grupo de duplicados repartido en varios conjuntos: {g}")
    return problems


def find_identical_files_across_splits(df, root, path_col="path", split_col="split"):
    """Detecta archivos identicos (mismo md5) que quedaron en conjuntos distintos."""
    root = Path(root)
    seen = {}
    problems = []
    for _, row in df.iterrows():
        f = root / row[path_col]
        if not f.exists():
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen and seen[h][1] != row[split_col]:
            problems.append(f"contenido identico en {seen[h][1]} y {row[split_col]}: {f}")
        seen.setdefault(h, (f, row[split_col]))
    return problems


def main():
    ap = argparse.ArgumentParser(description="Valida el split del dataset de maiz")
    ap.add_argument("--csv", required=True, help="CSV del split (path, label, split, group)")
    ap.add_argument("--root", required=True, help="carpeta que contiene las imagenes")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    root = Path(args.root)
    problems = validate_labels(df)
    problems += find_leakage(df)
    problems += validate_images([root / p for p in df["path"]])
    problems += find_identical_files_across_splits(df, root)

    for p in problems:
        print("X", p)
    print(f"{len(df)} filas revisadas, {len(problems)} problemas")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
