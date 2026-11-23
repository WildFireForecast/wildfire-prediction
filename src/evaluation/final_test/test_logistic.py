
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

MODEL_DIR = (
    ROOT / "models" / "logistic_regression" / "baseline"
)

DATA_DIR = (
    ROOT / "data" / "processed" / "provisional"
)

BASELINE_DIR = DATA_DIR / "baseline"

REPORT_DIR = ROOT / "reports" / "final_test"


# ==================================================
# 2. AUSGEWÄHLTE MODELLE
# ==================================================

MODELS = {
    "1A": {
        "path": MODEL_DIR / "logistic_1A.joblib",
        "features_key": "features_without_lst",
        "input_file": "test_X_without_lst.csv",
    },
    "1B": {
        "path": MODEL_DIR / "logistic_1B.joblib",
        "features_key": "features_with_lst",
        "input_file": "test_X_with_lst.csv",
    },
}


# ==================================================
# 3. MODELLE OHNE TESTDATEN PRÜFEN
# ==================================================

def check_models():

    print("=" * 60)
    print("LOGISTISCHE REGRESSION - MODELLPRÜFUNG")
    print("=" * 60)

    config = json.loads(
        (DATA_DIR / "input_config.json").read_text(
            encoding="utf-8"
        )
    )

    for name, settings in MODELS.items():

        print(f"\nPrüfe Modell {name} ...")

        model = joblib.load(settings["path"])

        expected_features = config[
            settings["features_key"]
        ]

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"{name}: predict_proba fehlt."
            )

        if model.n_features_in_ != len(expected_features):
            raise ValueError(
                f"{name}: Feature-Anzahl stimmt nicht."
            )

        if hasattr(model, "feature_names_in_"):
            if list(model.feature_names_in_) != expected_features:
                raise ValueError(
                    f"{name}: Feature-Reihenfolge stimmt nicht."
                )

        print("Modell geladen: OK")
        print("Features:", len(expected_features))
        print("Vorhersagefunktion vorhanden: OK")

    print("\nBeide Modelle erfolgreich geprüft.")
    print("Keine Testdaten verwendet.")


# ==================================================
# 4. FINALE TESTEVALUATION
# ==================================================

def run_final_evaluation():

    # WICHTIG:
    # Freigabe erfolgt VOR dem Laden der Testdaten.
    require_final_test_approval()

    if "--execute" not in sys.argv:
        raise RuntimeError(
            "Für die finale Testevaluation ist "
            "das Argument --execute erforderlich."
        )

    print("\nStarte finale Evaluation: 1A und 1B")

    config = json.loads(
        (DATA_DIR / "input_config.json").read_text(
            encoding="utf-8"
        )
    )

    # Zielwerte aller Testsamples
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
            "Test-Labels müssen binär sein."
        )

    if y.nunique() != 2:
        raise ValueError(
            "Beide Klassen müssen vorhanden sein."
        )

    all_results = []
    reference_ids = y.index

    for name, settings in MODELS.items():

        print(f"\nEvaluierung Modell {name} ...")

        model = joblib.load(settings["path"])

        expected_features = config[
            settings["features_key"]
        ]

        X = pd.read_csv(
            BASELINE_DIR / settings["input_file"],
            index_col="sample_id",
        ).sort_index()

        # Sample-Zuordnung sicherstellen
        if X.empty or X.index.has_duplicates:
            raise ValueError(
                f"{name}: Ungültige Sample-IDs."
            )

        if not X.index.equals(reference_ids):
            raise ValueError(
                f"{name}: Test-Sample-IDs stimmen nicht."
            )

        # Feature-Reihenfolge sicherstellen
        if list(X.columns) != expected_features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )

        if model.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"{name}: Feature-Anzahl stimmt nicht."
            )

        if hasattr(model, "feature_names_in_"):
            if list(model.feature_names_in_) != list(X.columns):
                raise ValueError(
                    f"{name}: Modell-Features stimmen nicht."
                )

        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError(
                f"{name}: Unendliche Feature-Werte."
            )

        # Keine neue Anpassung der Vorverarbeitung:
        # Bereits trainierte Pipeline verwenden.
        probabilities = model.predict_proba(X)[:, 1]

        metrics = evaluate_predictions(
            name,
            y.to_numpy(),
            probabilities,
        )

        all_results.append(metrics)

        predictions = (
            np.asarray(probabilities) >= 0.5
        ).astype(int)

        # Individuelle Vorhersagen speichern
        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_df = pd.DataFrame(
            {
                "label": y.to_numpy(),
                "score": probabilities,
                "prediction": predictions,
            },
            index=reference_ids,
        )

        prediction_df.index.name = "sample_id"

        prediction_df.to_csv(
            REPORT_DIR / f"{name}_test_predictions.csv"
        )

        print(
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )
        print(
            f"F1-Score: {metrics['f1']:.4f}"
        )

    # Gemeinsame Ergebnistabelle
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "logistic_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("LOGISTISCHE REGRESSION - TEST ABGESCHLOSSEN")
    print("=" * 60)

    print(results_df.to_string(index=False))
    print("\nErgebnisse gespeichert unter:")
    print(REPORT_DIR)


# ==================================================
# 5. PROGRAMMSTART
# ==================================================

def main():

    # Ohne --execute werden nur Modelle geprüft.
    if "--execute" not in sys.argv:
        check_models()
        return

    # Mit --execute greift die Freigabeprüfung.
    run_final_evaluation()


if __name__ == "__main__":
    main()
