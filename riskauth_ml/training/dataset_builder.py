""" Build a training feature matrix from stored auth events."""

import os
import sqlite3
from datetime import datetime

from riskauth_ml.features.extractor import extract_features
from riskauth_ml.features.baseline_store import InMemoryBaseline


def load_events_from_db(db_path: str | None = None) -> list[dict]:
    """Load events from the v2 auth_events table, ordered by time."""
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "riskauth-mlops", "events.db"
        )

    conn = sqlite3.connect(db_path)

    # Ensure the new schema table exists (no-op if it already does)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            tenant_id TEXT,
            ip_address TEXT,
            country TEXT,
            city TEXT,
            latitude REAL,
            longitude REAL,
            asn TEXT,
            user_agent TEXT,
            device_type TEXT,
            rtt REAL,
            login_timestamp TEXT,
            login_success INTEGER,
            risk_score REAL,
            decision TEXT,
            created_at TEXT
        )
    """)
    conn.commit()

    rows = conn.execute("""
        SELECT user_id, tenant_id, ip_address, country, city,
               latitude, longitude, asn, user_agent, device_type,
               rtt, login_timestamp, login_success
        FROM auth_events
        ORDER BY login_timestamp ASC
    """).fetchall()
    conn.close()

    events = []
    for r in rows:
        events.append({
            "user_id": r[0],
            "tenant_id": r[1],
            "ip_address": r[2],
            "country": r[3] or "Unknown",
            "city": r[4] or "Unknown",
            "latitude": r[5] or 0.0,
            "longitude": r[6] or 0.0,
            "asn": r[7] or "Unknown",
            "user_agent": r[8] or "Unknown",
            "device_type": r[9] or "desktop",
            "rtt": r[10] or 0.0,
            "login_timestamp": r[11],
            "login_success": bool(r[12]) if r[12] is not None else True,
        })
    return events


def build_feature_matrix(events: list[dict]) -> list[list[float]]:
    baseline_store = InMemoryBaseline()
    X: list[list[float]] = []

    for event in events:
        user_id = event["user_id"]

        # Get current baseline BEFORE this event
        baseline = baseline_store.get(user_id)

        # Extract features
        vector = extract_features(event, baseline)
        X.append(vector)

        # Update baseline AFTER feature extraction
        baseline_store.update(user_id, event)

    return X
