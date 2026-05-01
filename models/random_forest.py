"""models/random_forest.py - Random Forest model"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import config


def train(X_train, y_train) -> RandomForestClassifier:
    """Fit a Random Forest and return it."""
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        criterion=config.RF_CRITERION,
        max_depth=config.RF_MAX_DEPTH,
        max_features=config.RF_MAX_FEATURES,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def predict(model: RandomForestClassifier, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary predictions, probability scores)."""
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score