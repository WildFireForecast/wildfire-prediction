
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
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

# ==========================================
# 1. Projektpfade
# ==========================================

ROOT = Path(__file__).resolve().parents[4]

DATA = ROOT / "data/processed/provisional"
BASELINE = DATA / "baseline"
FEATURE_2B = DATA / "xgboost_2B"

MODELS = ROOT / "models/tuned"
REPORTS = ROOT / "reports/tuning_xgboost"

REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Daten laden
# ==========================================

def load_validation(variant):

    if variant == "2A":
        X = pd.read_csv(
            BASELINE / "validation_X_with_lst.csv",
            index_col="sample_id",
        )
    else:
        X = pd.read_csv(
            FEATURE_2B / "validation_X.csv",
            index_col="sample_id",
        )

    y = pd.read_csv(
        BASELINE / "validation_y.csv",
        index_col="sample_id",
    )["label"]

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError("Doppelte Sample-IDs.")

    if set(X.index) != set(y.index):
        raise ValueError("Sample-IDs stimmen nicht überein.")

    X = X.sort_index()
    y = y.loc[X.index]

    if X.empty or y.isna().any():
        raise ValueError("Ungültige Validierungsdaten.")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Beide Klassen müssen vorhanden sein.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Unendliche Eingabewerte.")

    return X, y


# ==========================================
# 3. Modelle evaluieren
# ==========================================

results = []

for variant in ["2A", "2B"]:

    print(f"\n=== EVALUATION MODELL {variant} ===")

    X_val, y_val = load_validation(variant)

    model_file = (
        MODELS / f"xgboost_{variant}_tuned.joblib"
    )

    model = joblib.load(model_file)

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(X_val.columns):
            raise ValueError(
                f"{variant}: Feature-Reihenfolge stimmt nicht."
            )

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": variant,
        "version": "tuned",
        "features": X_val.shape[1],
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_val, probabilities
        ),
        "average_precision": average_precision_score(
            y_val, probabilities
        ),
    }

    results.append(metrics)

    # Vorhersagen speichern
    pd.DataFrame(
        {
            "label": y_val,
            "score": probabilities,
            "prediction": predictions,
        },
        index=X_val.index,
    ).to_csv(
        REPORTS / f"{variant}_tuned_validation_predictions.csv"
    )

    # Verwechslungsmatrix speichern
    matrix = pd.DataFrame(
        confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ),
        index=["tatsaechlich_0", "tatsaechlich_1"],
        columns=["vorhergesagt_0", "vorhergesagt_1"],
    )

    matrix.to_csv(
        REPORTS / f"{variant}_tuned_confusion_matrix.csv"
    )

    print("\nVerwechslungsmatrix:")
    print(matrix.to_string())

# ==========================================
# 4. Ergebnisse zusammenfassen
# ==========================================

tuned = pd.DataFrame(results)

tuned.to_csv(
    REPORTS / "tuned_validation_metrics.csv",
    index=False,
)

# Bisherige Ergebnisse laden
original_2A = pd.read_csv(
    ROOT / "reports/xgboost_provisional/validation_metrics.csv"
)
original_2A["version"] = "baseline"

original_2B = pd.read_csv(
    ROOT / "reports/xgboost_2B_provisional/validation_metrics.csv"
)
original_2B["version"] = "baseline"

comparison = pd.concat(
    [original_2A, original_2B, tuned],
    ignore_index=True,
)

comparison = comparison[
    [
        "model",
        "version",
        "features",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
]

comparison.to_csv(
    REPORTS / "comparison_baseline_vs_tuned.csv",
    index=False,
)

# ==========================================
# 5. Ergebnisse anzeigen
# ==========================================

print("\n======================================")
print(" XGBOOST: BASELINE VS. TUNED")
print("======================================")

print(
    comparison.round(4).to_string(index=False)
)

print("\nErgebnisse gespeichert unter:")
print(REPORTS)

print("\nTestdaten 2021–2022 wurden nicht verwendet.")
