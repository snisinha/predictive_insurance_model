# models/neural_network.py

import numpy as np
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