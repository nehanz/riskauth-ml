import joblib
import os
from riskauth_ml.features.schema import FEATURE_ORDER

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "if_model.joblib"
)

_model = joblib.load(MODEL_PATH)


def to_vector(features: dict) -> list:
    return [features[f] for f in FEATURE_ORDER]


def predict(features: dict) -> float:
    vector = to_vector(features)
    raw_score = -_model.score_samples([vector])[0]
    print("Using Isolation Forest model")

    # normalize roughly into 0–1 range
    return min(max(raw_score, 0.0), 1.0)
