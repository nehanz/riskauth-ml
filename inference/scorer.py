from features.extractor import extract_features
from models.model import predict

def score(event: dict) -> float:
    features = extract_features(event)
    return predict(features)
