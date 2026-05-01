"""models/decision_tree.py - Decision Tree model"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import config


def train(X_train, y_train) -> DecisionTreeClassifier:
    """Fit a Decision Tree and return it."""
    model = DecisionTreeClassifier(
        max_depth=config.DT_MAX_DEPTH,
        criterion=config.DT_CRITERION,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def predict(model: DecisionTreeClassifier, X) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary predictions, probability scores)."""
    y_pred  = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score