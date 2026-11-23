
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_predictions(model_name, y_true, y_score):
    """Berechnet einheitliche Kennzahlen für ein Modell."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Labels und Scores müssen eindimensional sein.")

    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("Ungültige Anzahl von Vorhersagen.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ungültige Klassenlabels.")

    if not np.isfinite(y_score).all():
        raise ValueError("Scores enthalten NaN oder Infinity.")

    if not ((y_score >= 0) & (y_score <= 1)).all():
        raise ValueError("Scores müssen zwischen 0 und 1 liegen.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC-AUC benötigt beide Klassen.")

    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(
            y_true, y_score
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
