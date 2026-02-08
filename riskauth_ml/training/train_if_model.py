import sqlite3
import joblib
from sklearn.ensemble import IsolationForest
from datetime import datetime
from riskauth_ml.features.extractor import extract_features
from riskauth_ml.features.schema import FEATURE_ORDER
import os


def to_vector(features: dict) -> list:
    return [features[f] for f in FEATURE_ORDER]

def load_events(db_path=None):
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "riskauth-mlops", "events.db")
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS auth_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        tenant_id TEXT,
        ip TEXT,
        device TEXT,
        login_time TEXT,
        risk_score REAL,
        decision TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    rows = conn.execute("""
        SELECT user_id, tenant_id, ip, device, login_time
        FROM auth_events
    """).fetchall()

    events = []
    for r in rows:
        events.append({
            "user_id": r[0],
            "tenant_id": r[1],
            "ip": r[2],
            "device": r[3],
            "login_time": datetime.fromisoformat(r[4])
        })
    return events


def main():
    events = load_events()
    features_list = [extract_features(e) for e in events]
    
    X = [[f["login_hour"], f["device_change"], f["failed_logins_24h"]] for f in features_list]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    # Save to shared models directory
    model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "if_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
