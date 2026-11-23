
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from metrics import evaluate_predictions
from test_guard import require_final_test_approval


# ==================================================
# 1. PROJEKTPFADE
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = ROOT / "models" / "xgboost" / "tuned"

DATA_DIR = ROOT / "data" / "processed" / "provisional"

BASELINE_DIR = DATA_DIR / "baseline"

FEATURE_DIR = DATA_DIR / "xgboost_2B"

REPORT_DIR = ROOT / "reports" / "final_test"


# ==================================================
# 2. AUSGEWÄHLTE MODELLE
# ==================================================

MODELS = {
    "2A": {
        "model_path": MODEL_DIR / "xgboost_2A_tuned.joblib",
        "data_dir": BASELINE_DIR,
        "train_file": "train_X_with_lst.csv",
        "test_file": "test_X_with_lst.csv",
    },
    "2B": {
        "model_path": MODEL_DIR / "xgboost_2B_tuned.joblib",
        "data_dir": FEATURE_DIR,
        "train_file": "train_X.csv",
        "test_file": "test_X.csv",
    },
}


# ==================================================
# 3. FEATURE-PRÜFUNG
# ==================================================

def get_expected_features(name, settings):

    # Nur Spaltennamen aus den Trainingsdaten lesen.
    # Keine Trainingsdaten oder Testdaten laden.
    columns = pd.read_csv(
        settings["data_dir"] / settings["train_file"],
        nrows=0,
        index_col="sample_id",
    ).columns.tolist()

    if not columns:
        raise ValueError(
            f"{name}: Keine Features gefunden."
        )

    return columns


def validate_model_features(name, model, features):

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{name}: predict_proba fehlt."
        )

    if model.n_features_in_ != len(features):
        raise ValueError(
            f"{name}: Falsche Feature-Anzahl."
        )

    trained_features = model.get_booster().feature_names

    if trained_features is not None:
        if list(trained_features) != features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )


# ==================================================
# 4. MODELLE OHNE TESTDATEN PRÜFEN
# ==================================================

def check_models():

    print("=" * 60)
    print("XGBOOST - MODELLPRÜFUNG")
    print("=" * 60)

    for name, settings in MODELS.items():

        print(f"\nPrüfe Modell {name} ...", flush=True)

        model = joblib.load(
            settings["model_path"]
        )

        features = get_expected_features(
            name, settings
        )

        validate_model_features(
            name, model, features
        )

        print("Modell geladen: OK")
        print("Feature-Reihenfolge: OK")
        print("Features:", len(features))
        print("Vorhersagefunktion vorhanden: OK")

    print("\nBeide XGBoost-Modelle erfolgreich geprüft.")
    print("Keine Testdaten verwendet.")


# ==================================================
# 5. FINALE TESTEVALUATION
# ==================================================

def run_final_evaluation():

    # Vor jeglichem Zugriff auf Testdaten prüfen.
    require_final_test_approval()

    if "--execute" not in sys.argv:
        raise RuntimeError(
            "Die Ausführung benötigt --execute."
        )

    print("\nStarte finale XGBoost-Testevaluation.")

    # Testlabels laden
    y = pd.read_csv(
        BASELINE_DIR / "test_y.csv",
        index_col="sample_id",
    )["label"].sort_index()

    if y.empty or y.index.has_duplicates:
        raise ValueError(
            "Ungültige Test-Sample-IDs."
        )

    if not y.isin([0, 1]).all():
        raise ValueError(
            "Testlabels sind nicht binär."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Beide Klassen müssen vorhanden sein."
        )

    expected_ids = y.index
    all_results = []

    # Vorhersagen zunächst im Speicher sammeln.
    all_predictions = {}

    for name, settings in MODELS.items():

        print(f"\nEvaluiere Modell {name} ...", flush=True)

        model = joblib.load(
            settings["model_path"]
        )

        expected_features = get_expected_features(
            name, settings
        )

        validate_model_features(
            name, model, expected_features
        )

        X = pd.read_csv(
            settings["data_dir"] / settings["test_file"],
            index_col="sample_id",
        ).sort_index()

        if X.empty or X.index.has_duplicates:
            raise ValueError(
                f"{name}: Ungültige Test-Sample-IDs."
            )

        if not X.index.equals(expected_ids):
            raise ValueError(
                f"{name}: Sample-IDs stimmen nicht überein."
            )

        if list(X.columns) != expected_features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )

        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError(
                f"{name}: Unendliche Feature-Werte."
            )

        # Bereits trainiertes Modell verwenden.
        # Keine erneute Anpassung oder Optimierung.
        scores = model.predict_proba(X)[:, 1]

        metrics = evaluate_predictions(
            name,
            y.to_numpy(),
            scores,
        )

        all_results.append(metrics)

        predictions = (
            np.asarray(scores) >= 0.5
        ).astype(int)

        prediction_df = pd.DataFrame(
            {
                "label": y.to_numpy(),
                "score": scores,
                "prediction": predictions,
            },
            index=expected_ids,
        )

        prediction_df.index.name = "sample_id"

        all_predictions[name] = prediction_df

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

    # Ergebnisse erst nach erfolgreicher Evaluation
    # beider Varianten speichern.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, predictions in all_predictions.items():
        predictions.to_csv(
            REPORT_DIR / f"{name}_test_predictions.csv"
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "xgboost_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("XGBOOST - TEST ABGESCHLOSSEN")
    print("=" * 60)

    print(results_df.to_string(index=False))
    print("\nGespeichert unter:", REPORT_DIR)


# ==================================================
# 6. PROGRAMMSTART
# ==================================================

def main():

    if "--execute" not in sys.argv:
        check_models()
        return

    run_final_evaluation()


if __name__ == "__main__":
    main()
