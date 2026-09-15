"""Model evaluation utilities for unsupervised IsolationForest."""

import numpy as np

from riskauth_ml.features.schema import FEATURE_ORDER


def evaluate_model(model, scaler, X_raw: np.ndarray):
    """Print evaluation metrics for the trained model."""
    X_scaled = scaler.transform(X_raw)

    # Raw scores (more negative → more anomalous)
    raw_scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)  # 1 = inlier, -1 = outlier

    # Normalise to 0-1 risk scores via sigmoid
    risk_scores = 1.0 / (1.0 + np.exp(raw_scores))

    n_outliers = int(np.sum(predictions == -1))
    n_inliers = int(np.sum(predictions == 1))
    total = len(X_raw)

    print("Model Evaluation (Unsupervised)")
    print(f"   Samples:        {total}")
    print(f"   Features:       {X_raw.shape[1]}")
    print(f"   Inliers:        {n_inliers} ({100 * n_inliers / total:.1f}%)")
    print(f"   Outliers:       {n_outliers} ({100 * n_outliers / total:.1f}%)")
    print()
    print("   Risk Score Distribution:")
    print(f"     Mean:         {risk_scores.mean():.4f}")
    print(f"     Std:          {risk_scores.std():.4f}")
    print(f"     Min:          {risk_scores.min():.4f}")
    print(f"     Max:          {risk_scores.max():.4f}")
    print(f"     P25:          {np.percentile(risk_scores, 25):.4f}")
    print(f"     P50 (Median): {np.percentile(risk_scores, 50):.4f}")
    print(f"     P75:          {np.percentile(risk_scores, 75):.4f}")
    print(f"     P95:          {np.percentile(risk_scores, 95):.4f}")
    print(f"     P99:          {np.percentile(risk_scores, 99):.4f}")

    # Per-feature statistics
    print()
    print("   Per-Feature Mean / Std:")
    for i, name in enumerate(FEATURE_ORDER):
        col = X_raw[:, i]
        print(f"     {name:30s}  mean={col.mean():8.4f}  std={col.std():8.4f}")
