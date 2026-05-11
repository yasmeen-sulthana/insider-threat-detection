import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def aggregate_user_meta_features(
    user_sequences: dict,
    ae_errors: np.ndarray,
    bilstm_scores: np.ndarray,
    user_labels: list,
) -> pd.DataFrame:
    """
    Aggregate sequence-level signals to user-level meta-features.

    For each user, computes statistics across all their windows:
      - Autoencoder:  mean, max, std of reconstruction error
      - BiLSTM:       mean, max, std of anomaly score
      - Derived:      temporal_variation, behavior_deviation

    Returns DataFrame indexed by user with 8+ meta-feature columns.
    """
    seq_df = pd.DataFrame({
        "user": user_labels,
        "ae_error": ae_errors,
        "bilstm_score": bilstm_scores,
    })

    agg = seq_df.groupby("user").agg(
        # Autoencoder features
        mean_ae_error=("ae_error", "mean"),
        max_ae_error=("ae_error", "max"),
        std_ae_error=("ae_error", "std"),
        # BiLSTM features
        mean_bilstm_score=("bilstm_score", "mean"),
        max_bilstm_score=("bilstm_score", "max"),
        std_bilstm_score=("bilstm_score", "std"),
        # Sequence count (more sequences = more data)
        n_sequences=("ae_error", "count"),
    ).reset_index()

    # Fill NaN std (users with 1 sequence)
    agg["std_ae_error"] = agg["std_ae_error"].fillna(0)
    agg["std_bilstm_score"] = agg["std_bilstm_score"].fillna(0)

    # Derived meta-features
    agg["temporal_variation"] = agg["std_ae_error"] + agg["std_bilstm_score"]
    agg["behavior_deviation"] = (
        agg["mean_ae_error"] * 0.4
        + agg["max_ae_error"] * 0.2
        + agg["mean_bilstm_score"] * 0.3
        + agg["max_bilstm_score"] * 0.1
    )

    logger.info(f"Meta-features computed for {len(agg)} users")
    return agg


# The feature columns used by the Random Forest
META_FEATURE_COLS = [
    "mean_ae_error", "max_ae_error", "std_ae_error",
    "mean_bilstm_score", "max_bilstm_score", "std_bilstm_score",
    "temporal_variation", "behavior_deviation",
]


def classify_users(meta_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Train a Random Forest classifier on the user-level meta-features
    and return per-user predictions.

    Labelling strategy (unsupervised → semi-supervised):
      1. Scale all meta-features to [0,1]
      2. Compute a composite anomaly score (weighted mean)
      3. Users above the 80th percentile composite are labelled "Threat"
      4. Train RF on these pseudo-labels
      5. Use RF's own predicted probabilities as final scores

    The RF learns non-linear decision boundaries across ALL meta-features,
    producing more nuanced and varied predictions than a simple threshold.

    Returns:
        result_df:      DataFrame with [user, prediction, score, label]
        model_accuracy: float — cross-validated accuracy of the RF model
    """
    X_raw = meta_df[META_FEATURE_COLS].fillna(0).values

    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    # Weighted composite anomaly score for pseudo-labelling
    weights = np.array([0.20, 0.10, 0.05, 0.25, 0.10, 0.05, 0.10, 0.15])
    composite = X @ weights
    composite = composite / composite.max() if composite.max() > 0 else composite

    # Pseudo-labels: top 20% are threats
    threshold = np.percentile(composite, 80)
    pseudo_labels = (composite >= threshold).astype(int)

    n_threat = int(pseudo_labels.sum())
    n_normal = len(pseudo_labels) - n_threat
    logger.info(f"Pseudo-labels: {n_normal} normal, {n_threat} threat (threshold={threshold:.4f})")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, pseudo_labels)

    # Cross-validated accuracy (if enough samples)
    if len(X) >= 10:
        cv_folds = min(5, n_threat, n_normal)
        if cv_folds >= 2:
            cv_scores = cross_val_score(rf, X, pseudo_labels, cv=cv_folds, scoring="accuracy")
            model_accuracy = float(np.mean(cv_scores))
        else:
            model_accuracy = float(rf.score(X, pseudo_labels))
    else:
        model_accuracy = float(rf.score(X, pseudo_labels))

    # Final predictions using the trained model
    predictions = rf.predict(X)
    proba = rf.predict_proba(X)

    # Get probability of class 1 (threat)
    if proba.shape[1] == 2:
        threat_proba = proba[:, 1]
    else:
        threat_proba = proba[:, 0]

    # Feature importances
    importances = dict(zip(META_FEATURE_COLS, rf.feature_importances_))
    logger.info(f"RF feature importances: {importances}")

    result = meta_df[["user"]].copy()
    result["prediction"] = predictions
    result["score"] = np.round(threat_proba, 4)
    result["label"] = result["prediction"].map({0: "Normal", 1: "Threat"})

    logger.info(
        f"Classification complete: {int(predictions.sum())} threats / "
        f"{len(predictions)} users | CV accuracy={model_accuracy:.4f}"
    )
    return result, model_accuracy


def build_summary(result_df: pd.DataFrame, model_accuracy: float) -> dict:
    """Build the final JSON summary for the frontend."""
    total = len(result_df)
    threat_count = int(result_df["prediction"].sum())
    normal_count = total - threat_count

    user_results = []
    for _, row in result_df.iterrows():
        user_results.append({
            "user_id": str(row["user"]),
            "prediction": int(row["prediction"]),
            "label": row["label"],
            "score": float(row["score"]),
        })

    # Sort: threats first, then by score descending
    user_results.sort(key=lambda x: (-x["prediction"], -x["score"]))

    return {
        "total_users": total,
        "normal_users": normal_count,
        "threat_users": threat_count,
        "model_accuracy": round(model_accuracy, 4),
        "user_results": user_results,
    }
