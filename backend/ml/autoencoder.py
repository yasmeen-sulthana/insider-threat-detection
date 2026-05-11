import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_autoencoder(seq_len: int, n_features: int):
    """
    Build an LSTM-based Autoencoder for unsupervised anomaly detection.

    Architecture:
        Encoder: LSTM(64) → LSTM(32) → bottleneck
        Decoder: RepeatVector → LSTM(32) → LSTM(64) → Dense(n_features)

    Trained to reconstruct input sequences; high reconstruction error
    indicates anomalous behaviour.
    """
    import tensorflow as tf
    from tensorflow import keras

    # Suppress TF info logs
    tf.get_logger().setLevel("ERROR")

    inp = keras.Input(shape=(seq_len, n_features), name="ae_input")

    # Encoder
    enc = keras.layers.LSTM(64, activation="relu", return_sequences=True, name="enc_lstm1")(inp)
    enc = keras.layers.LSTM(32, activation="relu", return_sequences=False, name="enc_lstm2")(enc)

    # Bottleneck
    bottleneck = keras.layers.RepeatVector(seq_len, name="bottleneck")(enc)

    # Decoder
    dec = keras.layers.LSTM(32, activation="relu", return_sequences=True, name="dec_lstm1")(bottleneck)
    dec = keras.layers.LSTM(64, activation="relu", return_sequences=True, name="dec_lstm2")(dec)
    out = keras.layers.TimeDistributed(keras.layers.Dense(n_features), name="ae_output")(dec)

    model = keras.Model(inp, out, name="LSTM_Autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_autoencoder(
    X: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    validation_split: float = 0.1,
) -> np.ndarray:
    """
    Train the Autoencoder on ALL sequences and compute reconstruction error.

    The model learns to reconstruct normal-looking sequences well.
    Anomalous sequences produce higher reconstruction error.

    Args:
        X: (N, seq_len, n_features) — all user sequences

    Returns:
        errors: (N,) — per-sequence mean squared reconstruction error
    """
    if X.shape[0] == 0:
        return np.array([], dtype=np.float32)

    seq_len, n_features = X.shape[1], X.shape[2]
    model = build_autoencoder(seq_len, n_features)

    logger.info(f"Training Autoencoder on {X.shape[0]} sequences, "
                f"shape=({seq_len},{n_features}), epochs={epochs}")

    model.fit(
        X, X,
        epochs=epochs,
        batch_size=min(batch_size, X.shape[0]),
        validation_split=validation_split if X.shape[0] > 20 else 0.0,
        shuffle=True,
        verbose=0,
    )

    # Reconstruct and compute per-sequence MSE
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=(1, 2)).astype(np.float64)

    logger.info(
        f"Autoencoder errors — "
        f"min={errors.min():.6f}, median={np.median(errors):.6f}, "
        f"max={errors.max():.6f}, std={errors.std():.6f}"
    )
    return errors
