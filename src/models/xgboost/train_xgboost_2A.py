
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost

from xgboost import XGBClassifier
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

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data/processed/provisional/baseline"
MODELS = ROOT / "models/provisional"
REPORTS = ROOT / "reports/xgboost_provisional"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Konfiguration laden
# ==========================================

config = json.loads(
    (DATA.parent / "input_config.json").read_text(
        encoding="utf-8"
    )
)

features = config["features_with_lst"]

# ==========================================
# 3. Trainings- und Validierungsdaten laden
# ==========================================

def load_data(split):
    X = pd.read_csv(
        DATA / f"{split}_X_with_lst.csv",
        index_col="sample_id",
    )

    y = pd.read_csv(
        DATA / f"{split}_y.csv",
        index_col="sample_id",
    )["label"]

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError(f"{split}: Doppelte Sample-IDs.")

    if set(X.index) != set(y.index):
        raise ValueError(
            f"{split}: Samples stimmen nicht überein."
        )

    if list(X.columns) != features:
        raise ValueError(
            f"{split}: Unerwartete Eingabemerkmale."
        )

    X = X.sort_index()
    y = y.loc[X.index]

    if X.empty:
        raise ValueError(f"{split}: Keine Daten vorhanden.")

    if X.isna().all(axis=0).any():
        raise ValueError(
            f"{split}: Mindestens ein Merkmal ist vollständig leer."
        )

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{split}: Ungültige Labels.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(
            f"{split}: Unendliche Werte vorhanden."
        )

    return X, y


X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")

if not X_train.index.intersection(X_val.index).empty:
    raise ValueError(
        "Überschneidung zwischen Training und Validierung."
    )

print("\n=== MODELL 2A: XGBOOST ===")
print("Training:", X_train.shape)
print("Validierung:", X_val.shape)
print("Anzahl Features:", len(features))

print("\nKlassenverteilung Training:")
print(y_train.value_counts().sort_index())

print("\nKlassenverteilung Validierung:")
print(y_val.value_counts().sort_index())

# ==========================================
# 4. XGBoost-Modell definieren
# ==========================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

# ==========================================
# 5. Modell trainieren
# ==========================================

print("\nTrainiere Modell 2A ...", flush=True)

model.fit(X_train, y_train)

print("Training erfolgreich abgeschlossen.")

# ==========================================
# 6. Validierung 2020
# ==========================================

probabilities = model.predict_proba(X_val)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "model": "2A",
    "features": len(features),
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

summary = pd.DataFrame(
    [metrics]
).set_index("model")

# ==========================================
# 7. Modell und Ergebnisse speichern
# ==========================================

joblib.dump(
    model,
    MODELS / "xgboost_2A.joblib",
)

summary.to_csv(
    REPORTS / "validation_metrics.csv"
)

pd.DataFrame(
    {
        "label": y_val,
        "score": probabilities,
        "prediction": predictions,
    },
    index=X_val.index,
).to_csv(
    REPORTS / "2A_validation_predictions.csv"
)

matrix = pd.DataFrame(
    confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ),
    index=["tatsaechlich_0", "tatsaechlich_1"],
    columns=["vorhergesagt_0", "vorhergesagt_1"],
)

matrix.to_csv(
    REPORTS / "2A_confusion_matrix.csv"
)

run_info = {
    "model": "2A",
    "status": "provisional",
    "algorithm": "XGBoost",
    "xgboost_version": xgboost.__version__,
    "feature_engineering": False,
    "features": features,
    "input_time_index": 58,
    "target_day_assumption": config["target_day_assumption"],
    "training_years": "2006-2019",
    "validation_year": 2020,
    "test_used": False,
    "threshold": 0.5,
    "missing_values": "Native XGBoost missing-value handling",
    "parameters": model.get_params(),
}

(REPORTS / "run_info.json").write_text(
    json.dumps(
        run_info,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

# ==========================================
# 8. Ergebnisse anzeigen
# ==========================================

print("\n=== VALIDIERUNGSERGEBNISSE 2020 ===")
print(summary.round(4).to_string())

print("\n=== VERWECHSLUNGSMATRIX ===")
print(matrix.to_string())

print("\nModell gespeichert:")
print(MODELS / "xgboost_2A.joblib")

print("\nErgebnisse gespeichert:")
print(REPORTS)

print("\nTestdaten wurden nicht verwendet.")
