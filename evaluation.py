# evaluation.py — metrics, confusion matrix, ROC curve, and model comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from typing import Dict, Any


def _save(fig_name: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, fig_name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    output_dir: str,
) -> Dict[str, float]:
    """
    Print classification report, plot confusion matrix and ROC curve.

    Parameters
    ----------
    name       : human-readable model label (e.g. "Random Forest")
    y_true     : ground-truth labels
    y_pred     : predicted binary labels
    y_score    : predicted probabilities / decision scores for ROC
    output_dir : directory to save plot files

    Returns
    -------
    dict with train_acc, test_acc, sensitivity, specificity, roc_auc
    """
    test_acc  = accuracy_score(y_true, y_pred) * 100
    cm        = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(classification_report(y_true, y_pred))
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Confusion matrix heatmap
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save(f"cm_{name.replace(' ', '_').lower()}.png", output_dir)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)
    plt.figure()
    plt.title(f"{name} — ROC Curve")
    plt.plot(fpr, tpr, color="green", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save(f"roc_{name.replace(' ', '_').lower()}.png", output_dir)

    return {
        "test_acc":    test_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc":     roc_auc,
    }


def compare_models(results: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Bar chart comparing train vs test accuracy for all models,
    and a summary DataFrame.
    """
    labels     = list(results.keys())
    train_accs = [results[m].get("train_acc", 0) for m in labels]
    test_accs  = [results[m]["test_acc"] for m in labels]

    n     = len(labels)
    r     = np.arange(n)
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(r,         train_accs, color="b", width=width, edgecolor="black", label="Train acc")
    plt.bar(r + width, test_accs,  color="g", width=width, edgecolor="black", label="Test acc")
    plt.xticks(r + width / 2, labels)
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    _save("model_comparison.png", output_dir)

    summary = pd.DataFrame(results).T
    print("\n── Model Summary ───────────────────────────────────────────────")
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, "model_summary.csv"))
    print(f"\nSummary saved to {output_dir}/model_summary.csv")