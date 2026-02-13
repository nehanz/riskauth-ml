""" RiskAuth Training Pipeline v2 """

import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from riskauth_ml.features.schema import FEATURE_ORDER, FEATURE_COUNT
from riskauth_ml.training.dataset_builder import (
    load_events_from_db,
    build_feature_matrix,
)
from riskauth_ml.training.evaluate import evaluate_model


def train(db_path: str | None = None, model_dir: str | None = None):
    """Full training pipeline."""

    # 1. Load events
    print("[1/5] Loading events from database...")
    events = load_events_from_db(db_path)

    if len(events) < 10:
        print(f"⚠  Only {len(events)} events found. Need at least 10 for training.")
        print("   Run generate_traffic.py first to populate the database.")
        sys.exit(1)

    print(f"      → {len(events)} events loaded")

    # 2. Build feature matrix
    print("[2/5] Building 21-feature matrix...")
    X = build_feature_matrix(events)
    X = np.array(X, dtype=np.float64)

    # Handle NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"      → Matrix shape: {X.shape}")
    assert X.shape[1] == FEATURE_COUNT, (
        f"Expected {FEATURE_COUNT} features, got {X.shape[1]}"
    )

    # 3. Fit scaler
    print("[3/5] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train IsolationForest
    print("[4/5] Training IsolationForest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        max_samples="auto",
        random_state=42,
    )
    model.fit(X_scaled)

    # 5. Save artefacts
    if model_dir is None:
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "models"
        )
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "if_model_v2.joblib")
    scaler_path = os.path.join(model_dir, "scaler_v2.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("[5/5] Artefacts saved:")
    print(f"      → Model:  {model_path}")
    print(f"      → Scaler: {scaler_path}")

    # Evaluate
    print("\n" + "=" * 55)
    evaluate_model(model, scaler, X)
    print("=" * 55)
    print("Training complete.")


def main():
    train()


if __name__ == "__main__":
    main()
