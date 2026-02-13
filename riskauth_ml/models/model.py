""" Model loader — loads IsolationForest v2 + StandardScaler."""

import math
import os

import joblib

_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models"
)

MODEL_PATH = os.path.join(_MODELS_DIR, "if_model_v2.joblib")
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler_v2.joblib")


def _load_artifact(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


_model = _load_artifact(MODEL_PATH)
_scaler = _load_artifact(SCALER_PATH)


def predict(vector: list[float]) -> float:
    global _model, _scaler

    # Lazy re-check: allows hot-reloading after training
    if _model is None or _scaler is None:
        _model = _load_artifact(MODEL_PATH)
        _scaler = _load_artifact(SCALER_PATH)

    if _model is None or _scaler is None:
        # No model artefacts yet — return neutral score
        return 0.5

    scaled = _scaler.transform([vector])
    raw_score = -_model.decision_function(scaled)[0]

    # Sigmoid normalisation into 0–1
    normalized = 1.0 / (1.0 + math.exp(-raw_score))
    return round(min(max(normalized, 0.0), 1.0), 4)
