from datetime import datetime
import random

def extract_features(event: dict) -> dict:
    return {
        "login_hour": event["login_time"].hour,
        "device_change": 1 if event["device"] == "mobile" else 0,
        "failed_logins_24h": random.randint(0, 5)  # placeholder
    }
