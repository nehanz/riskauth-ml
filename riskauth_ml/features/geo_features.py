"""Geo-location feature computation for RiskAuth v2."""

import math
from typing import Optional


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in kilometres."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def country_change_flag(country: str, known_countries: list[str]) -> float:
    """1.0 if country has never been seen for this user."""
    if not known_countries:
        return 0.0  # first login — no anomaly
    return 0.0 if country in known_countries else 1.0


def city_change_flag(city: str, known_cities: list[str]) -> float:
    """1.0 if city has never been seen for this user."""
    if not known_cities:
        return 0.0
    return 0.0 if city in known_cities else 1.0


def geo_distance_from_last_km(
    lat: float, lon: float,
    last_lat: float, last_lon: float,
) -> float:
    """Distance from last login location in km."""
    if last_lat == 0.0 and last_lon == 0.0:
        return 0.0  # no previous location recorded
    return haversine_km(lat, lon, last_lat, last_lon)


def geo_velocity_score(
    distance_km: float,
    hours_since_last: Optional[float],
) -> float:
    """
    Impossible-travel indicator.
    Normalised: 0 = normal, 1 = physically impossible.
    Threshold: >900 km/h (commercial jet speed).
    """
    if hours_since_last is None or hours_since_last <= 0:
        return 0.0
    velocity = distance_km / hours_since_last
    return min(velocity / 900.0, 1.0)


def geo_frequency_score(country: str, country_freq: dict[str, int]) -> float:
    """
    How common is this location for the user.
    1.0 = most common, 0.0 = never seen.
    """
    if not country_freq:
        return 1.0  # first login
    total = sum(country_freq.values())
    count = country_freq.get(country, 0)
    return count / total if total > 0 else 0.0
