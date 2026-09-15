"""Device behaviour feature computation for RiskAuth v2."""

import math
from collections import Counter

from riskauth_ml.features.schema import DEVICE_TYPE_MAP


def device_change_flag(device_type: str, known_devices: list[str]) -> float:
    """1.0 if device type has never been seen for this user."""
    if not known_devices:
        return 0.0
    return 0.0 if device_type in known_devices else 1.0


def device_frequency_score(device_type: str, device_freq: dict[str, int]) -> float:
    """How common this device type is for the user."""
    if not device_freq:
        return 1.0
    total = sum(device_freq.values())
    count = device_freq.get(device_type, 0)
    return count / total if total > 0 else 0.0


def user_agent_entropy(user_agent: str) -> float:
    """Shannon entropy of user-agent string characters."""
    if not user_agent or user_agent == "Unknown":
        return 0.0
    counts = Counter(user_agent)
    length = len(user_agent)
    entropy = -sum(
        (c / length) * math.log2(c / length) for c in counts.values() if c > 0
    )
    return entropy


def device_type_encoded(device_type: str) -> float:
    """Encode device type: Desktop=0, Mobile=1, Bot=2."""
    return float(DEVICE_TYPE_MAP.get(device_type.lower(), 0))
