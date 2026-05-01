"""models/neural_network.py - Neural Network model"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import config


def build_model(input_dim: int) -> keras.Sequential:
    """Build and compile a fully-connected binary classifier."""
    layers = [keras.layers.Dense(config.NN_HIDDEN_LAYERS[0],
                                  input_shape=(input_dim,),
                                  activation=config.NN_ACTIVATION)]
    for units in config.NN_HIDDEN_LAYERS[1:]:
        layers.append(keras.layers.Dense(units, activation=config.NN_ACTIVATION))
    layers.append(keras.layers.Dense(1, activation=config.NN_OUTPUT_ACTIVATION))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=config.NN_OPTIMIZER,
        loss=config.NN_LOSS,
        metrics=["accuracy"],
    )
    return model


def train(X_train, y_train) -> keras.Sequential:
    """Fit the neural network and return the trained model."""
    model = build_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=config.NN_EPOCHS, verbose=1)
    return model


def predict(model: keras.Sequential, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary predictions, raw probability scores)."""
    y_score = model.predict(X).flatten()
    y_pred  = (y_score > config.NN_THRESHOLD).astype(int)
    return y_pred, y_score


def predict_to_csv(
    model: keras.Sequential,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: str,
    filename: str = "nn_test_predictions.csv",
) -> str:
    """
    Run the model on X_test, then save a CSV with columns:
        predicted_label  — 0 or 1
        probability      — raw sigmoid score
        actual_label     — ground truth (optional; pass None to omit)

    Returns the path to the saved file.
    """
    y_pred, y_score = predict(model, X_test)

    results = pd.DataFrame({
        "predicted_label": y_pred,
        "probability":     y_score.round(4),
    })

    if y_test is not None:
        results.insert(0, "actual_label", y_test)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}  ({len(results)} rows)")
    return out_path