FEATURE_ORDER = [
    # Time-Based Behaviour (4)
    "login_hour_sin",
    "login_hour_cos",
    "is_weekend",
    "login_hour_deviation",
    # Geo Behaviour (5)
    "country_change_flag",
    "city_change_flag",
    "geo_velocity_score",
    "geo_distance_from_last_km",
    "geo_frequency_score",
    # Network Behaviour (4)
    "asn_change_flag",
    "ip_reputation_score",
    "rtt_deviation",
    "network_frequency_score",
    # Device Behaviour (4)
    "device_change_flag",
    "device_frequency_score",
    "user_agent_entropy",
    "device_type_encoded",
    # Account Behaviour (4)
    "failed_logins_24h",
    "success_ratio_7d",
    "account_age_days",
    "login_attempt_rate_1h",
]

FEATURE_COUNT = len(FEATURE_ORDER)


DEVICE_TYPE_MAP = {
    "desktop": 0,
    "mobile": 1,
    "bot": 2,
}

DEFAULT_BASELINE = {
    "avg_login_hour": 12.0,
    "avg_rtt": 50.0,
    "known_countries": [],
    "known_cities": [],
    "known_asns": [],
    "known_devices": [],
    "success_ratio_7d": 1.0,
    "account_age_days": 0.0,
    "total_logins": 0,
    "failed_logins_24h": 0,
    "login_attempt_rate_1h": 0.0,
    "last_login_timestamp": None,
    "last_latitude": 0.0,
    "last_longitude": 0.0,
    "country_freq": {},
    "asn_freq": {},
    "device_freq": {},
    "first_login_timestamp": None,
}
