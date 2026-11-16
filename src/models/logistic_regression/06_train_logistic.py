from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "provisional" / "baseline"
MODELS = ROOT / "models" / "provisional"
REPORTS = ROOT / "reports" / "logistic_regression/baseline"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

config = json.loads(
    (DATA.parent / "input_config.json").read_text(encoding="utf-8")
)

# Bei nicht abgeschlossenem Optimierungsverfahren anhalten.
warnings.filterwarnings("error", category=ConvergenceWarning)


def load_data(split, variant, features):
    X = pd.read_csv(
        DATA / f"{split}_X_{variant}.csv",
        index_col="sample_id",
    )
    y = pd.read_csv(
        DATA / f"{split}_y.csv",
        index_col="sample_id",
    )["label"]

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError(f"{split}: Doppelte Sample-IDs.")

    if set(X.index) != set(y.index):
        raise ValueError(f"{split}: Eingaben und Labels passen nicht.")

    if list(X.columns) != features:
        raise ValueError(f"{split}: Unerwartete Eingabemerkmale.")

    X = X.sort_index()
    y = y.loc[X.index]

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError(f"{split}: Ungültige oder fehlende Klassen.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(f"{split}: Unendliche Messwerte vorhanden.")

    return X, y


results = []

variants = [
    ("1A", "without_lst", config["features_without_lst"]),
    ("1B", "with_lst", config["features_with_lst"]),
]

for model_name, variant, features in variants:
    X_train, y_train = load_data("train", variant, features)
    X_val, y_val = load_data("validation", variant, features)

    if not X_train.index.intersection(X_val.index).empty:
        raise ValueError("Samples sind in Training und Validierung enthalten.")

    empty_columns = X_train.columns[X_train.isna().all()].tolist()
    if empty_columns:
        raise ValueError(
            f"Vollständig fehlende Trainingsmerkmale: {empty_columns}"
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
        )),
    ])

    print(f"\nTrainiere Modell {model_name} ...", flush=True)

    # Median, Skalierung und Modell nur aus Trainingsdaten lernen.
    model.fit(X_train, y_train)

    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_val)[:, positive_column]

    # Gleicher, vorab festgelegter Schwellenwert für beide Varianten.
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "model": model_name,
        "features": len(features),
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "average_precision": average_precision_score(y_val, probabilities),
    })

    joblib.dump(model, MODELS / f"logistic_{model_name}.joblib")

    pd.DataFrame({
        "label": y_val,
        "score": probabilities,
        "prediction": predictions,
    }, index=X_val.index).to_csv(
        REPORTS / f"{model_name}_validation_predictions.csv"
    )

    matrix = pd.DataFrame(
        confusion_matrix(y_val, predictions, labels=[0, 1]),
        index=["tatsaechlich_0", "tatsaechlich_1"],
        columns=["vorhergesagt_0", "vorhergesagt_1"],
    )
    matrix.to_csv(REPORTS / f"{model_name}_confusion_matrix.csv")

    print(f"Modell {model_name}: Verwechslungsmatrix")
    print(matrix.to_string())


summary = pd.DataFrame(results).set_index("model")
summary.to_csv(REPORTS / "validation_metrics.csv")

run_info = {
    "status": "provisional",
    "target_day_assumption": config["target_day_assumption"],
    "verification": config["verification"],
    "sklearn_version": sklearn.__version__,
    "training_years": "2006–2019",
    "validation_year": 2020,
    "test_used": False,
    "threshold": 0.5,
    "C": 1.0,
    "imputation": "Trainingsmedian pro Merkmal, beide Klassen gemeinsam",
    "scaling": "StandardScaler, ausschliesslich auf Training angepasst",
}

(REPORTS / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n--- VORLÄUFIGE ERGEBNISSE: VALIDIERUNG 2020 ---")
print(summary.round(4).to_string())
print("\nModelle: models/logistic_regression/baseline/")
print("Ergebnisse: reports/logistic_regression/baseline/")
print("Testdaten wurden nicht verwendet.")
