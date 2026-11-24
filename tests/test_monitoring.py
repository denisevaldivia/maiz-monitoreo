from src.monitoring.logger import log_prediction, summarize


def test_log_and_summary(tmp_path, monkeypatch):
    f = tmp_path / "pred.jsonl"
    monkeypatch.setenv("PREDICTION_LOG", str(f))
    base = {"probabilities": {}, "n_patches": 8}
    log_prediction(
        {"predicted_class": "healthy", "confidence": 0.9, "needs_review": False, **base}, 12.0
    )
    log_prediction(
        {"predicted_class": "common_rust", "confidence": 0.5, "needs_review": True, **base}, 15.0
    )
    s = summarize()
    assert s["n"] == 2
    assert s["review_rate"] == 0.5
    assert s["by_class"] == {"healthy": 1, "common_rust": 1}


def test_summary_without_log(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "nada.jsonl"))
    assert summarize() == {"n": 0}
