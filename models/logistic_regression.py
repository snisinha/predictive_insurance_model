# models/logistic_regression.py

import numpy as np
from sklearn.linear_model import LogisticRegression
import config


def train(X_train, y_train) -> LogisticRegression:
    """Fit a Logistic Regression model and return it."""
    model = LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary predictions, probability scores)."""
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score