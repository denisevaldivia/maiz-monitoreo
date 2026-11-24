"""API de inferencia: GET /health y POST /predict.

Correr:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src import config
from src.data.preprocessing import InvalidImageError, load_image
from src.models.loader import load_predictor
from src.monitoring.logger import log_prediction

log = logging.getLogger("maize.api")


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = None
    try:
        app.state.predictor = load_predictor()
    except FileNotFoundError as e:
        log.warning("Modelo no cargado: %s", e)
    yield


app = FastAPI(title="Maize leaf classifier", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="El modelo aun no esta cargado.")

    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Maximo {config.MAX_UPLOAD_MB} MB.")
    try:
        img = load_image(data)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    t0 = time.perf_counter()
    result = app.state.predictor.predict(img)
    latency_ms = (time.perf_counter() - t0) * 1000
    log_prediction(result, latency_ms, filename=file.filename)

    if result["needs_review"]:
        result["message"] = "Confianza baja: se recomienda revision de un agronomo."
    return result
