def predict(features: dict) -> float:
    score = 0.2
    score += features["failed_logins_24h"] * 0.1
    score += features["device_change"] * 0.3
    return min(score, 1.0)
