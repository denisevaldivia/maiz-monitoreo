import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.main import app
from src.models.loader import build_model
from src.models.predictor import Predictor


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API con un modelo de pesos aleatorios (no hace falta el archivo real)."""
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "pred.jsonl"))
    with TestClient(app) as c:
        app.state.predictor = Predictor(build_model())
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_image(client, good_image_bytes):
    r = client.post("/predict", files={"file": ("hoja.jpg", good_image_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in config.CLASSES
    assert set(body["probabilities"]) == set(config.CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-2)
    assert isinstance(body["needs_review"], bool)


def test_predict_invalid_file_returns_clear_error(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"no soy imagen", "image/jpeg")})
    assert r.status_code == 400
    assert "imagen" in r.json()["detail"].lower()


def test_predict_without_file_is_422(client):
    assert client.post("/predict").status_code == 422


def test_predict_without_model_is_503(client, good_image_bytes):
    app.state.predictor = None
    r = client.post("/predict", files={"file": ("hoja.jpg", good_image_bytes, "image/jpeg")})
    assert r.status_code == 503
