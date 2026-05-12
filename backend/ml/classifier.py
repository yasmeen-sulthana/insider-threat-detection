import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import skfuzzy as fuzz

logger = logging.getLogger(__name__)

def pcm_cluster(X, c, m=2.0, eta=None, max_iter=100, error=1e-5):
    n_samples, n_features = X.shape
    np.random.seed(42)
    centers = X[np.random.choice(n_samples, c, replace=False)]
    
    if eta is None:
        dist_to_mean = np.linalg.norm(X - X.mean(axis=0), axis=1)
        eta = np.ones(c) * np.mean(dist_to_mean)
        
    u = np.zeros((c, n_samples))
    for i in range(max_iter):
        u_old = u.copy()
        
        d = np.zeros((c, n_samples))
        for j in range(c):
            d[j] = np.linalg.norm(X - centers[j], axis=1) ** 2
            
        for j in range(c):
            d_safe = np.maximum(d[j], 1e-10)
            u[j] = 1.0 / (1.0 + (d_safe / eta[j]) ** (1.0 / (m - 1)))
            
        um = u ** m
        for j in range(c):
            centers[j] = np.sum(X * um[j, :, np.newaxis], axis=0) / np.maximum(np.sum(um[j]), 1e-10)
            
        if np.linalg.norm(u - u_old) < error:
            break
            
    return centers, u


def aggregate_user_meta_features(
    user_sequences: dict,
    ae_errors: np.ndarray,
    bilstm_scores: np.ndarray,
    user_labels: list,
) -> pd.DataFrame:
    seq_df = pd.DataFrame({
        "user": user_labels,
        "ae_error": ae_errors,
        "bilstm_score": bilstm_scores,
    })

    agg = seq_df.groupby("user").agg(
        mean_ae_error=("ae_error", "mean"),
        max_ae_error=("ae_error", "max"),
        std_ae_error=("ae_error", "std"),
        mean_bilstm_score=("bilstm_score", "mean"),
        max_bilstm_score=("bilstm_score", "max"),
        std_bilstm_score=("bilstm_score", "std"),
        n_sequences=("ae_error", "count"),
    ).reset_index()

    agg["std_ae_error"] = agg["std_ae_error"].fillna(0)
    agg["std_bilstm_score"] = agg["std_bilstm_score"].fillna(0)

    agg["temporal_variation"] = agg["std_ae_error"] + agg["std_bilstm_score"]
    agg["behavior_deviation"] = (
        agg["mean_ae_error"] * 0.4
        + agg["max_ae_error"] * 0.2
        + agg["mean_bilstm_score"] * 0.3
        + agg["max_bilstm_score"] * 0.1
    )

    logger.info(f"Meta-features computed for {len(agg)} users")
    return agg


META_FEATURE_COLS = [
    "mean_ae_error", "max_ae_error", "std_ae_error",
    "mean_bilstm_score", "max_bilstm_score", "std_bilstm_score",
    "temporal_variation", "behavior_deviation",
]


def classify_users(meta_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    X_raw = meta_df[META_FEATURE_COLS].fillna(0).values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    weights = np.array([0.20, 0.10, 0.05, 0.25, 0.10, 0.05, 0.10, 0.15])
    composite = X @ weights
    composite = composite / composite.max() if composite.max() > 0 else composite

    threshold = np.percentile(composite, 80)
    pseudo_labels = (composite >= threshold).astype(int)

    n_threat = int(pseudo_labels.sum())
    n_normal = len(pseudo_labels) - n_threat
    logger.info(f"Pseudo-labels: {n_normal} normal, {n_threat} threat (threshold={threshold:.4f})")

    cv_folds = min(5, n_threat, n_normal) if len(X) >= 10 else 0

    accuracies = {}
    models_predictions = {}
    models_proba = {}

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X, pseudo_labels)
    if cv_folds >= 2:
        accuracies["Random Forest"] = float(np.mean(cross_val_score(rf, X, pseudo_labels, cv=cv_folds, scoring="accuracy")))
    else:
        accuracies["Random Forest"] = float(rf.score(X, pseudo_labels))
    models_predictions["Random Forest"] = rf.predict(X)
    models_proba["Random Forest"] = rf.predict_proba(X)[:, 1] if rf.predict_proba(X).shape[1] == 2 else rf.predict_proba(X)[:, 0]

    # 2. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)
    lr.fit(X, pseudo_labels)
    if cv_folds >= 2:
        accuracies["Logistic Regression"] = float(np.mean(cross_val_score(lr, X, pseudo_labels, cv=cv_folds, scoring="accuracy")))
    else:
        accuracies["Logistic Regression"] = float(lr.score(X, pseudo_labels))
    models_predictions["Logistic Regression"] = lr.predict(X)
    models_proba["Logistic Regression"] = lr.predict_proba(X)[:, 1] if lr.predict_proba(X).shape[1] == 2 else lr.predict_proba(X)[:, 0]

    # 3. XGBoost
    try:
        xgb_model = xgb.XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
        xgb_model.fit(X, pseudo_labels)
        if cv_folds >= 2:
            accuracies["XGBoost"] = float(np.mean(cross_val_score(xgb_model, X, pseudo_labels, cv=cv_folds, scoring="accuracy")))
        else:
            accuracies["XGBoost"] = float(xgb_model.score(X, pseudo_labels))
        models_predictions["XGBoost"] = xgb_model.predict(X)
        proba_xgb = xgb_model.predict_proba(X)
        models_proba["XGBoost"] = proba_xgb[:, 1] if proba_xgb.shape[1] == 2 else proba_xgb[:, 0]
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        accuracies["XGBoost"] = 0.0
        models_predictions["XGBoost"] = np.zeros(len(X))
        models_proba["XGBoost"] = np.zeros(len(X))

    # 4. Fuzzy C-Means (FCM)
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(X.T, c=2, m=2, error=0.005, maxiter=1000, init=None)
        cluster_labels = np.argmax(u, axis=0)
        mean_comp_0 = np.mean(composite[cluster_labels == 0]) if np.sum(cluster_labels == 0) > 0 else 0
        mean_comp_1 = np.mean(composite[cluster_labels == 1]) if np.sum(cluster_labels == 1) > 0 else 0
        
        threat_cluster_idx = 1 if mean_comp_1 > mean_comp_0 else 0
        fcm_predictions = (cluster_labels == threat_cluster_idx).astype(int)
        fcm_proba = u[threat_cluster_idx]
        
        accuracies["Fuzzy C-Means"] = float(accuracy_score(pseudo_labels, fcm_predictions))
        models_predictions["Fuzzy C-Means"] = fcm_predictions
        models_proba["Fuzzy C-Means"] = fcm_proba
    except Exception as e:
        logger.error(f"FCM failed: {e}")
        accuracies["Fuzzy C-Means"] = 0.0
        models_predictions["Fuzzy C-Means"] = np.zeros(len(X))
        models_proba["Fuzzy C-Means"] = np.zeros(len(X))

    # 5. Possibilistic C-Means (PCM)
    try:
        centers, u_pcm = pcm_cluster(X, c=2, m=2.0)
        cluster_labels_pcm = np.argmax(u_pcm, axis=0)
        
        mean_comp_0_pcm = np.mean(composite[cluster_labels_pcm == 0]) if np.sum(cluster_labels_pcm == 0) > 0 else 0
        mean_comp_1_pcm = np.mean(composite[cluster_labels_pcm == 1]) if np.sum(cluster_labels_pcm == 1) > 0 else 0
        
        threat_cluster_idx_pcm = 1 if mean_comp_1_pcm > mean_comp_0_pcm else 0
        pcm_predictions = (cluster_labels_pcm == threat_cluster_idx_pcm).astype(int)
        pcm_proba = u_pcm[threat_cluster_idx_pcm]
        pcm_proba = np.clip(pcm_proba, 0, 1)
        
        accuracies["PCM"] = float(accuracy_score(pseudo_labels, pcm_predictions))
        models_predictions["PCM"] = pcm_predictions
        models_proba["PCM"] = pcm_proba
    except Exception as e:
        logger.error(f"PCM failed: {e}")
        accuracies["PCM"] = 0.0
        models_predictions["PCM"] = np.zeros(len(X))
        models_proba["PCM"] = np.zeros(len(X))

    # Pick best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    
    result = meta_df[["user"]].copy()
    result["prediction"] = models_predictions[best_model_name]
    result["score"] = np.round(models_proba[best_model_name], 4)
    result["label"] = result["prediction"].map({0: "Normal", 1: "Threat"})

    logger.info(f"Classification complete. Best Model: {best_model_name} with ACC={best_accuracy:.4f}")
    return result, accuracies, best_model_name


def build_summary(result_df: pd.DataFrame, accuracies: dict, best_model_name: str) -> dict:
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

    user_results.sort(key=lambda x: (-x["prediction"], -x["score"]))

    best_accuracy = round(float(accuracies.get(best_model_name, 0)), 4)
    model_accuracies = {k: round(float(v), 4) for k, v in accuracies.items()}

    return {
        "total_users": total,
        "normal_users": normal_count,
        "threat_users": threat_count,
        "model_accuracy": best_accuracy,
        "best_model": best_model_name,
        "model_accuracies": model_accuracies,
        "user_results": user_results,
    }
