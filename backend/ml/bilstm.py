import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_bilstm(seq_len: int, n_features: int):
    """
    Build a Bidirectional LSTM for temporal sequence behaviour scoring.

    Architecture:
        BiLSTM(64) → BiLSTM(32) → Dense(32, relu) → Dropout → Dense(1, sigmoid)

    Trained with anomaly-aware pseudo-labels derived from Autoencoder
    reconstruction errors (passed in during training).
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel("ERROR")

    inp = keras.Input(shape=(seq_len, n_features), name="bilstm_input")

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, activation="tanh", return_sequences=True),
        name="bilstm_1"
    )(inp)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(32, activation="tanh", return_sequences=False),
        name="bilstm_2"
    )(x)
    x = keras.layers.Dense(32, activation="relu", name="fc1")(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="bilstm_output")(x)

    model = keras.Model(inp, out, name="BiLSTM_Scorer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_bilstm(
    X: np.ndarray,
    ae_errors: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> np.ndarray:
    """
    Train BiLSTM using Autoencoder reconstruction errors as pseudo-labels.

    This creates an anomaly-aware temporal model:
      - Sequences with high AE reconstruction error are labelled as anomalous
      - The BiLSTM learns temporal patterns that distinguish normal vs anomalous

    Args:
        X:          (N, seq_len, n_features) — all user sequences
        ae_errors:  (N,) — per-sequence reconstruction errors from Autoencoder

    Returns:
        scores: (N,) — per-sequence behaviour anomaly scores in [0, 1]
    """
    if X.shape[0] == 0:
        return np.array([], dtype=np.float32)

    seq_len, n_features = X.shape[1], X.shape[2]

    # Derive pseudo-labels from AE errors:
    # sequences above the 80th percentile error are "anomalous"
    error_threshold = np.percentile(ae_errors, 80)
    pseudo_labels = (ae_errors >= error_threshold).astype(np.float32)

    n_anomalous = int(pseudo_labels.sum())
    n_normal = len(pseudo_labels) - n_anomalous
    logger.info(
        f"BiLSTM pseudo-labels: {n_normal} normal, {n_anomalous} anomalous "
        f"(threshold={error_threshold:.6f})"
    )

    model = build_bilstm(seq_len, n_features)

    logger.info(f"Training BiLSTM on {X.shape[0]} sequences, epochs={epochs}")

    model.fit(
        X, pseudo_labels,
        epochs=epochs,
        batch_size=min(batch_size, X.shape[0]),
        validation_split=validation_split if X.shape[0] > 20 else 0.0,
        shuffle=True,
        verbose=0,
    )

    # Predict anomaly scores
    scores = model.predict(X, batch_size=batch_size, verbose=0).flatten().astype(np.float64)

    logger.info(
        f"BiLSTM scores — "
        f"min={scores.min():.6f}, median={np.median(scores):.6f}, "
        f"max={scores.max():.6f}, std={scores.std():.6f}"
    )
    return scores
