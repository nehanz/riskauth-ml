import random

def extract_features(event: dict) -> dict:
    return {
        "failed_logins_24h": random.randint(0, 5),
        "device_change": random.choice([0, 1]),
        "hour": event["login_time"].hour
    }
