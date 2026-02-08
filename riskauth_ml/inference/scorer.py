from riskauth_ml.features.extractor import extract_features
from riskauth_ml.models.model import predict


def score(event: dict) -> float:
    features = extract_features(event)
    return predict(features)
