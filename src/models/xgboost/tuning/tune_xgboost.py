
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

# ==========================================
# 1. Projektpfade
# ==========================================

ROOT = Path(__file__).resolve().parents[4]

DATA = ROOT / "data/processed/provisional"
BASELINE = DATA / "baseline"
FEATURE_2B = DATA / "xgboost_2B"

MODELS = ROOT / "models/xgboost/tuned"
REPORTS = ROOT / "reports/xgboost/tuning"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Hyperparameter-Suchraum
# ==========================================

PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

N_ITER = 20
SEED = 42

# ==========================================
# 3. Beide Varianten definieren
# ==========================================

VARIANTS = {
    "2A": BASELINE,
    "2B": FEATURE_2B,
}

def load_data(variant, split):
    folder = VARIANTS[variant]

    if variant == "2A":
        filename = f"{split}_X_with_lst.csv"
    else:
        filename = f"{split}_X.csv"

    X = pd.read_csv(
        folder / filename,
        index_col="sample_id",
    ).sort_index()

    y = pd.read_csv(
        BASELINE / f"{split}_y.csv",
        index_col="sample_id",
    )["label"].sort_index()

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError("Doppelte Sample-IDs.")

    if not X.index.equals(y.index):
        raise ValueError("Sample-IDs stimmen nicht überein.")

    if X.empty or X.isna().all(axis=0).any():
        raise ValueError("Leere Daten oder Merkmale.")

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Ungültige Zielwerte.")

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Unendliche Eingabewerte.")

    return X, y

# ==========================================
# 4. Hyperparameter-Tuning
# ==========================================

for variant in VARIANTS:

    print(f"\n=== XGBOOST TUNING MODELL {variant} ===")

    X_train, y_train = load_data(variant, "train")
    X_val, y_val = load_data(variant, "validation")

    if not X_train.columns.equals(X_val.columns):
        raise ValueError("Feature-Spalten stimmen nicht überein.")

    if not X_train.index.intersection(X_val.index).empty:
        raise ValueError("Training und Validierung überschneiden sich.")

    print("Training:", X_train.shape)
    print("Validierung:", X_val.shape)

    combinations = list(
        ParameterSampler(
            PARAM_GRID,
            n_iter=N_ITER,
            random_state=SEED,
        )
    )

    results = []
    best_score = -np.inf
    best_model = None
    best_params = None

    for i, params in enumerate(combinations, start=1):

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probabilities)

        results.append({
            "iteration": i,
            **params,
            "roc_auc": float(auc),
        })

        print(
            f"{variant} | Durchlauf {i:02d}/{N_ITER} "
            f"| ROC-AUC: {auc:.4f}",
            flush=True,
        )

        if auc > best_score:
            best_score = auc
            best_model = model
            best_params = params.copy()

    # ======================================
    # 5. Ergebnisse speichern
    # ======================================

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )

    results_df.to_csv(
        REPORTS / f"{variant}_tuning_results.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        MODELS / f"xgboost_{variant}_tuned.joblib",
    )

    info = {
        "model": variant,
        "status": "provisional",
        "selection_metric": "roc_auc",
        "best_roc_auc_validation": float(best_score),
        "best_parameters": best_params,
        "search_iterations": N_ITER,
        "random_state": SEED,
        "training_years": "2006-2019",
        "validation_year": 2020,
        "target_day_assumption": "time_idx=59 (unverified)",
        "test_used": False,
    }

    (REPORTS / f"{variant}_best_params.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n=== BESTES ERGEBNIS {variant} ===")
    print("ROC-AUC:", round(best_score, 4))
    print("Parameter:", best_params)
    print("Modell und Ergebnisse gespeichert.")

print("\n=== XGBOOST TUNING ABGESCHLOSSEN ===")
print("Testdaten wurden nicht verwendet.")
