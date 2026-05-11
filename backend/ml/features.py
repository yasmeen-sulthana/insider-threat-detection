import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

SEQ_LEN = 7
FEATURE_COLS = [
    "activity_count", "logon_count", "device_count",
    "file_count", "email_count", "http_count"
]


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    MinMax-normalize all feature columns to [0, 1].
    Missing feature columns are created as 0.
    """
    df = df.copy()
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    scaler = MinMaxScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].fillna(0).values)

    logger.info(f"Normalized {len(FEATURE_COLS)} features across {len(df)} rows")
    return df, scaler


def create_sequences(
    df: pd.DataFrame,
) -> tuple[dict, np.ndarray, list]:
    """
    Create fixed-length sliding-window sequences per user, sorted by date.

    For each user:
      - Sort rows by date
      - Extract feature matrix
      - Create overlapping windows of length SEQ_LEN
      - If user has fewer rows than SEQ_LEN, zero-pad at the front

    Returns:
        user_sequences : {user_id: ndarray (n_windows, SEQ_LEN, n_features)}
        X_all          : stacked ndarray (total_windows, SEQ_LEN, n_features)
        user_labels    : list of user_id per window, aligned with X_all
    """
    user_sequences = {}
    X_list = []
    user_labels = []

    for user, group in df.sort_values("date").groupby("user"):
        data = group[FEATURE_COLS].values.astype(np.float32)
        n = len(data)

        # Zero-pad if fewer rows than SEQ_LEN
        if n < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - n, len(FEATURE_COLS)), dtype=np.float32)
            data = np.vstack([pad, data])
            n = SEQ_LEN

        # Sliding windows
        windows = []
        for i in range(n - SEQ_LEN + 1):
            windows.append(data[i : i + SEQ_LEN])

        seq = np.array(windows, dtype=np.float32)
        user_sequences[user] = seq
        X_list.append(seq)
        user_labels.extend([user] * len(seq))

    X_all = np.vstack(X_list) if X_list else np.empty(
        (0, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32
    )

    logger.info(
        f"Sequences created: {X_all.shape[0]} windows from "
        f"{len(user_sequences)} users  |  shape={X_all.shape}"
    )
    return user_sequences, X_all, user_labels
