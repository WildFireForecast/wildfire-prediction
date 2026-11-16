
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

DATA = ROOT / "data/processed/provisional"
FEATURE_DATA = DATA / "xgboost_2B"
BASELINE = DATA / "baseline"

MODELS = ROOT / "models/xgboost/baseline"
REPORTS = ROOT / "reports/xgboost/baseline_2B"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Feature-Konfiguration laden
# ==========================================

config = json.loads(
    (FEATURE_DATA / "feature_config.json").read_text(
        encoding="utf-8"
    )
)

features = (
    config["original_features"]
    + list(config["engineered_features"].keys())
)

if len(features) != 41 or len(set(features)) != 41:
    raise ValueError("Erwartet werden 41 eindeutige Features.")

# ==========================================
# 3. Daten laden und prüfen
# ==========================================

def load_data(split):
    X = pd.read_csv(
        FEATURE_DATA / f"{split}_X.csv",
        index_col="sample_id",
    )

    y = pd.read_csv(
        BASELINE / f"{split}_y.csv",
        index_col="sample_id",
    )["label"]

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError(f"{split}: Doppelte Sample-IDs.")

    if set(X.index) != set(y.index):
        raise ValueError(f"{split}: Sample-IDs stimmen nicht überein.")

    if list(X.columns) != features:
        raise ValueError(f"{split}: Unerwartete Feature-Liste.")

    X = X.sort_index()
    y = y.loc[X.index]

    if X.empty or X.isna().all(axis=0).any():
        raise ValueError(f"{split}: Leere Daten oder Merkmale.")

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{split}: Ungültige Labels.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{split}: Unendliche Werte.")

    return X, y


X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")

if not X_train.index.intersection(X_val.index).empty:
    raise ValueError("Überschneidung zwischen Train und Validation.")

# Prüfen: dieselben 27 ursprünglichen Features wie 2A
for split, X in [("train", X_train), ("validation", X_val)]:
    X_2A = pd.read_csv(
        BASELINE / f"{split}_X_with_lst.csv",
        index_col="sample_id",
    ).sort_index()

    pd.testing.assert_frame_equal(
        X[config["original_features"]],
        X_2A,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-9,
    )

print("\n=== MODELL 2B: XGBOOST MIT FEATURE ENGINEERING ===")
print("Training:", X_train.shape)
print("Validierung:", X_val.shape)
print("Originalfeatures:", len(config["original_features"]))
print("Neue Features:", len(config["engineered_features"]))
print("Features insgesamt:", len(features))

# ==========================================
# 4. Modell definieren
# ==========================================

# Identische Hyperparameter wie Modell 2A
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
# 5. Training
# ==========================================

print("\nTrainiere Modell 2B ...", flush=True)

model.fit(X_train, y_train)

print("Training erfolgreich abgeschlossen.")

# ==========================================
# 6. Validierung
# ==========================================

probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "model": "2B",
    "features": len(features),
    "accuracy": accuracy_score(y_val, predictions),
    "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
    "precision": precision_score(y_val, predictions, zero_division=0),
    "recall": recall_score(y_val, predictions, zero_division=0),
    "f1": f1_score(y_val, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_val, probabilities),
    "average_precision": average_precision_score(
        y_val, probabilities
    ),
}

summary = pd.DataFrame([metrics]).set_index("model")

# ==========================================
# 7. Ergebnisse speichern
# ==========================================

joblib.dump(
    model,
    MODELS / "xgboost_2B.joblib",
)

summary.to_csv(
    REPORTS / "validation_metrics.csv"
)

pd.DataFrame({
    "label": y_val,
    "score": probabilities,
    "prediction": predictions,
}, index=X_val.index).to_csv(
    REPORTS / "2B_validation_predictions.csv"
)

matrix = pd.DataFrame(
    confusion_matrix(y_val, predictions, labels=[0, 1]),
    index=["tatsaechlich_0", "tatsaechlich_1"],
    columns=["vorhergesagt_0", "vorhergesagt_1"],
)

matrix.to_csv(
    REPORTS / "2B_confusion_matrix.csv"
)

run_info = {
    "model": "2B",
    "status": "provisional",
    "algorithm": "XGBoost",
    "xgboost_version": xgboost.__version__,
    "feature_engineering": True,
    "features": features,
    "input_indices": [0, 58],
    "target_day_assumption": config["target_day_assumption"],
    "training_years": "2006-2019",
    "validation_year": 2020,
    "test_used": False,
    "threshold": 0.5,
    "missing_values": "Native XGBoost missing-value handling",
    "parameters": model.get_params(),
}

(REPORTS / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# ==========================================
# 8. Vergleich mit Modell 2A
# ==========================================

metrics_2A = pd.read_csv(
    ROOT / "reports/xgboost/baseline_2A/validation_metrics.csv",
    index_col="model",
)

comparison = pd.concat([metrics_2A, summary])

comparison.to_csv(
    REPORTS / "comparison_2A_2B.csv"
)

print("\n=== VALIDIERUNGSERGEBNISSE 2020 ===")
print(summary.round(4).to_string())

print("\n=== VERWECHSLUNGSMATRIX ===")
print(matrix.to_string())

print("\n=== VERGLEICH MODELL 2A VS. 2B ===")
print(comparison.round(4).to_string())

print("\nModell gespeichert:")
print(MODELS / "xgboost_2B.joblib")

print("\nErgebnisse gespeichert:")
print(REPORTS)

print("\nTestdaten wurden nicht verwendet.")
