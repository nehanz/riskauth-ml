""" Stateless inference scorer for RiskAuth v2."""

from riskauth_ml.features.extractor import extract_features
from riskauth_ml.models.model import predict


def score(event: dict, baseline: dict) -> float:
    """Score a login event against the user's baseline."""
    vector = extract_features(event, baseline)
    return predict(vector)
