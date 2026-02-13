"""
Per-user baseline store for RiskAuth v2.

Two modes:
  - InMemoryBaseline : used during training (no DB dependency)
  - SQLiteBaseline   : used during online inference (reads/writes DB)
"""

import json
import sqlite3
from copy import deepcopy
from datetime import datetime
from typing import Optional

from riskauth_ml.features.schema import DEFAULT_BASELINE

# In-Memory Store (training)

class InMemoryBaseline:
    """In-memory per-user baseline for the training pipeline."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, user_id: str) -> dict:
        if user_id not in self._store:
            self._store[user_id] = deepcopy(DEFAULT_BASELINE)
        return deepcopy(self._store[user_id])

    def update(self, user_id: str, event: dict) -> None:
        bl = self._store.setdefault(user_id, deepcopy(DEFAULT_BASELINE))
        _update_baseline(bl, event)


# SQLite-Backed Store (online inference)

class SQLiteBaseline:
    """SQLite-backed per-user baseline for online inference."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_baselines (
                user_id TEXT PRIMARY KEY,
                avg_login_hour REAL DEFAULT 12.0,
                avg_rtt REAL DEFAULT 50.0,
                known_countries TEXT DEFAULT '[]',
                known_cities TEXT DEFAULT '[]',
                known_asns TEXT DEFAULT '[]',
                known_devices TEXT DEFAULT '[]',
                success_ratio_7d REAL DEFAULT 1.0,
                account_age_days REAL DEFAULT 0.0,
                total_logins INTEGER DEFAULT 0,
                failed_logins_24h INTEGER DEFAULT 0,
                login_attempt_rate_1h REAL DEFAULT 0.0,
                last_login_timestamp TEXT,
                last_latitude REAL DEFAULT 0.0,
                last_longitude REAL DEFAULT 0.0,
                country_freq TEXT DEFAULT '{}',
                asn_freq TEXT DEFAULT '{}',
                device_freq TEXT DEFAULT '{}',
                first_login_timestamp TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def get(self, user_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM user_baselines WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return deepcopy(DEFAULT_BASELINE)
        return _row_to_baseline(row)

    def update(self, user_id: str, event: dict) -> None:
        bl = self.get(user_id)
        _update_baseline(bl, event)
        self._upsert(user_id, bl, event)

    def _upsert(self, user_id: str, bl: dict, event: dict) -> None:
        now = datetime.utcnow().isoformat()
        first_ts = bl.get("first_login_timestamp") or event.get("login_timestamp", now)
        if hasattr(first_ts, "isoformat"):
            first_ts = first_ts.isoformat()

        self.conn.execute(
            """
            INSERT INTO user_baselines (
                user_id, avg_login_hour, avg_rtt,
                known_countries, known_cities, known_asns, known_devices,
                success_ratio_7d, account_age_days, total_logins,
                failed_logins_24h, login_attempt_rate_1h,
                last_login_timestamp, last_latitude, last_longitude,
                country_freq, asn_freq, device_freq,
                first_login_timestamp, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                avg_login_hour=excluded.avg_login_hour,
                avg_rtt=excluded.avg_rtt,
                known_countries=excluded.known_countries,
                known_cities=excluded.known_cities,
                known_asns=excluded.known_asns,
                known_devices=excluded.known_devices,
                success_ratio_7d=excluded.success_ratio_7d,
                account_age_days=excluded.account_age_days,
                total_logins=excluded.total_logins,
                failed_logins_24h=excluded.failed_logins_24h,
                login_attempt_rate_1h=excluded.login_attempt_rate_1h,
                last_login_timestamp=excluded.last_login_timestamp,
                last_latitude=excluded.last_latitude,
                last_longitude=excluded.last_longitude,
                country_freq=excluded.country_freq,
                asn_freq=excluded.asn_freq,
                device_freq=excluded.device_freq,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                bl["avg_login_hour"],
                bl["avg_rtt"],
                json.dumps(bl["known_countries"]),
                json.dumps(bl.get("known_cities", [])),
                json.dumps(bl["known_asns"]),
                json.dumps(bl["known_devices"]),
                bl["success_ratio_7d"],
                bl["account_age_days"],
                bl["total_logins"],
                bl["failed_logins_24h"],
                bl.get("login_attempt_rate_1h", 0.0),
                bl.get("last_login_timestamp"),
                bl.get("last_latitude", 0.0),
                bl.get("last_longitude", 0.0),
                json.dumps(bl.get("country_freq", {})),
                json.dumps(bl.get("asn_freq", {})),
                json.dumps(bl.get("device_freq", {})),
                first_ts,
                now,
            ),
        )
        self.conn.commit()

# Shared helpers

def _row_to_baseline(row) -> dict:
    """Convert a SQLite row tuple to a baseline dict."""
    return {
        "avg_login_hour": row[1] or 12.0,
        "avg_rtt": row[2] or 50.0,
        "known_countries": json.loads(row[3] or "[]"),
        "known_cities": json.loads(row[4] or "[]"),
        "known_asns": json.loads(row[5] or "[]"),
        "known_devices": json.loads(row[6] or "[]"),
        "success_ratio_7d": row[7] or 1.0,
        "account_age_days": row[8] or 0.0,
        "total_logins": row[9] or 0,
        "failed_logins_24h": row[10] or 0,
        "login_attempt_rate_1h": row[11] or 0.0,
        "last_login_timestamp": row[12],
        "last_latitude": row[13] or 0.0,
        "last_longitude": row[14] or 0.0,
        "country_freq": json.loads(row[15] or "{}"),
        "asn_freq": json.loads(row[16] or "{}"),
        "device_freq": json.loads(row[17] or "{}"),
        "first_login_timestamp": row[18],
    }


def _update_baseline(bl: dict, event: dict) -> None:
    """Incrementally update a baseline dict from a new event."""
    n = bl["total_logins"]

    # Parse login hour
    ts = event.get("login_timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    hour = ts.hour if ts else 12.0

    # Running average login hour
    bl["avg_login_hour"] = (bl["avg_login_hour"] * n + hour) / (n + 1)

    # Running average RTT
    rtt = event.get("rtt", 0.0)
    bl["avg_rtt"] = (bl["avg_rtt"] * n + rtt) / (n + 1)

    # Known sets
    country = event.get("country", "Unknown")
    city = event.get("city", "Unknown")
    asn = event.get("asn", "Unknown")
    device = event.get("device_type", "desktop")

    if country not in bl["known_countries"]:
        bl["known_countries"].append(country)
    if "known_cities" not in bl:
        bl["known_cities"] = []
    if city not in bl["known_cities"]:
        bl["known_cities"].append(city)
    if asn not in bl["known_asns"]:
        bl["known_asns"].append(asn)
    if device not in bl["known_devices"]:
        bl["known_devices"].append(device)

    # Frequency dicts
    cf = bl.setdefault("country_freq", {})
    cf[country] = cf.get(country, 0) + 1
    af = bl.setdefault("asn_freq", {})
    af[asn] = af.get(asn, 0) + 1
    df = bl.setdefault("device_freq", {})
    df[device] = df.get(device, 0) + 1

    # Success ratio (approximate rolling window)
    success = event.get("login_success", True)
    if isinstance(success, int):
        success = bool(success)
    old_ratio = bl["success_ratio_7d"]
    window = min(n, 50)  # approximate 7-day window
    bl["success_ratio_7d"] = (old_ratio * window + (1.0 if success else 0.0)) / (window + 1)

    # Account age
    first_ts = bl.get("first_login_timestamp")
    if first_ts and ts:
        if isinstance(first_ts, str):
            first_ts_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        else:
            first_ts_dt = first_ts
        try:
            bl["account_age_days"] = max((ts - first_ts_dt).total_seconds() / 86400, 0.0)
        except Exception:
            bl["account_age_days"] = 0.0
    elif not first_ts and ts:
        bl["first_login_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    # Update last location
    bl["last_latitude"] = event.get("latitude", 0.0)
    bl["last_longitude"] = event.get("longitude", 0.0)
    bl["last_login_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    # Increment total
    bl["total_logins"] = n + 1
