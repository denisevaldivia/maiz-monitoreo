"""Registro de predicciones (JSONL) y resumen simple para monitoreo."""

import json
import os
import time
from collections import Counter
from pathlib import Path

from src import config


def log_path():
    return Path(os.getenv("PREDICTION_LOG", config.ROOT / "logs" / "predictions.jsonl"))


def log_prediction(result, latency_ms, filename=None):
    """Agrega una linea al log. No guarda la imagen, solo el resultado."""
    f = log_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "filename": filename,
        "latency_ms": round(latency_ms, 1),
        **result,
    }
    with f.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def summarize(path=None):
    """Total de predicciones, % para revision, confianza media y conteo por clase."""
    f = Path(path) if path else log_path()
    if not f.exists():
        return {"n": 0}
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "review_rate": sum(r["needs_review"] for r in rows) / len(rows),
        "mean_confidence": sum(r["confidence"] for r in rows) / len(rows),
        "by_class": dict(Counter(r["predicted_class"] for r in rows)),
    }


def upload_log_to_s3(bucket, key, path=None):
    """Sube el log a S3 (se puede correr con cron en la EC2)."""
    import boto3

    boto3.client("s3").upload_file(str(path or log_path()), bucket, key)
