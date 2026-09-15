""" RiskAuth Feature Extractor v2 — 21-dimensional numeric vector."""

import math
from datetime import datetime
from typing import Optional

from riskauth_ml.features.schema import FEATURE_ORDER, FEATURE_COUNT
from riskauth_ml.features import geo_features, network_features, device_features


def extract_features(event: dict, baseline: dict) -> list[float]:
    """
    Extract a fixed-length 21-dimensional feature vector.

    Parameters
    ----------
    event : dict
        Raw login event with keys:
            login_timestamp, country, city, latitude, longitude,
            asn, user_agent, device_type, rtt, login_success
    baseline : dict
        Per-user historical baseline (from baseline store).

    Returns
    -------
    list[float]  — length == 21, order matches FEATURE_ORDER
    """
    # Parse timestamp
    ts = event.get("login_timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    hour = ts.hour if ts else 12.0
    weekday = ts.weekday() if ts else 0  # 0=Mon … 6=Sun

    # 1. Time features (4)
    login_hour_sin = math.sin(2 * math.pi * hour / 24)
    login_hour_cos = math.cos(2 * math.pi * hour / 24)
    is_weekend = 1.0 if weekday >= 5 else 0.0
    login_hour_deviation = abs(hour - baseline.get("avg_login_hour", 12.0))

    # 2. Geo features (5)
    country = event.get("country", "Unknown")
    city = event.get("city", "Unknown")
    lat = event.get("latitude", 0.0)
    lon = event.get("longitude", 0.0)

    _country_change = geo_features.country_change_flag(
        country, baseline.get("known_countries", [])
    )
    _city_change = geo_features.city_change_flag(
        city, baseline.get("known_cities", [])
    )
    dist_km = geo_features.geo_distance_from_last_km(
        lat, lon,
        baseline.get("last_latitude", 0.0),
        baseline.get("last_longitude", 0.0),
    )
    # hours since last login
    hours_since: Optional[float] = None
    last_ts = baseline.get("last_login_timestamp")
    if last_ts and ts:
        if isinstance(last_ts, str):
            last_ts_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        else:
            last_ts_dt = last_ts
        try:
            hours_since = max((ts - last_ts_dt).total_seconds() / 3600, 0.001)
        except Exception:
            hours_since = None

    _geo_velocity = geo_features.geo_velocity_score(dist_km, hours_since)
    _geo_freq = geo_features.geo_frequency_score(
        country, baseline.get("country_freq", {})
    )

    # 3. Network features (4)
    asn = event.get("asn", "Unknown")
    rtt = event.get("rtt", 0.0)

    _asn_change = network_features.asn_change_flag(
        asn, baseline.get("known_asns", [])
    )
    _ip_rep = network_features.ip_reputation_score()
    _rtt_dev = network_features.rtt_deviation(rtt, baseline.get("avg_rtt", 50.0))
    _net_freq = network_features.network_frequency_score(
        asn, baseline.get("asn_freq", {})
    )

    # 4. Device features (4)
    dev_type = event.get("device_type", "desktop")
    ua = event.get("user_agent", "Unknown")

    _dev_change = device_features.device_change_flag(
        dev_type, baseline.get("known_devices", [])
    )
    _dev_freq = device_features.device_frequency_score(
        dev_type, baseline.get("device_freq", {})
    )
    _ua_entropy = device_features.user_agent_entropy(ua)
    _dev_encoded = device_features.device_type_encoded(dev_type)

    # 5. Account features (4)
    _failed_24h = float(baseline.get("failed_logins_24h", 0))
    _success_7d = float(baseline.get("success_ratio_7d", 1.0))
    _acct_age = float(baseline.get("account_age_days", 0.0))
    _rate_1h = float(baseline.get("login_attempt_rate_1h", 0.0))

    # Assemble vector (order MUST match FEATURE_ORDER)
    vector = [
        login_hour_sin,
        login_hour_cos,
        is_weekend,
        login_hour_deviation,
        _country_change,
        _city_change,
        _geo_velocity,
        dist_km,
        _geo_freq,
        _asn_change,
        _ip_rep,
        _rtt_dev,
        _net_freq,
        _dev_change,
        _dev_freq,
        _ua_entropy,
        _dev_encoded,
        _failed_24h,
        _success_7d,
        _acct_age,
        _rate_1h,
    ]

    assert len(vector) == FEATURE_COUNT, (
        f"Feature vector length {len(vector)} != expected {FEATURE_COUNT}"
    )
    return vector
