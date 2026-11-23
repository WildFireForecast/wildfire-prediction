
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import evaluate_predictions


# ==================================================
# 1. PROJEKTPFADE
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = ROOT / "reports/final_test"

OUTPUT_FILE = (
    REPORT_DIR / "final_test_comparison.csv"
)

EXPECTED_SAMPLES = 775


# ==================================================
# 2. MODELLDEFINITIONEN
# ==================================================

MODELS = {
    "1A": {
        "family": "Logistische Regression",
        "version": "baseline",
        "features": 25,
        "source": "logistic_test_metrics.csv",
    },
    "1B": {
        "family": "Logistische Regression",
        "version": "baseline",
        "features": 27,
        "source": "logistic_test_metrics.csv",
    },
    "2A": {
        "family": "XGBoost",
        "version": "tuned",
        "features": 27,
        "source": "xgboost_test_metrics.csv",
    },
    "2B": {
        "family": "XGBoost",
        "version": "tuned",
        "features": 41,
        "source": "xgboost_test_metrics.csv",
    },
    "3A": {
        "family": "LSTM",
        "version": "baseline",
        "features": 25,
        "source": "lstm_test_metrics.csv",
    },
    "3B": {
        "family": "LSTM",
        "version": "tuned",
        "features": 27,
        "source": "lstm_test_metrics.csv",
    },
    "4A": {
        "family": "Transformer",
        "version": "tuned",
        "features": 25,
        "source": "transformer_test_metrics.csv",
    },
    "4B": {
        "family": "Transformer",
        "version": "tuned",
        "features": 27,
        "source": "transformer_test_metrics.csv",
    },
}


# ==================================================
# 3. GESPEICHERTE KENNZAHLEN LADEN
# ==================================================

def load_saved_metrics(model_name, settings):

    path = REPORT_DIR / settings["source"]

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    selected = df.loc[
        df["model"].astype(str).eq(model_name)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"{model_name}: Erwartet wird genau "
            "eine gespeicherte Ergebniszeile."
        )

    return selected.iloc[0]


# ==================================================
# 4. VORHERSAGEN VALIDIEREN
# ==================================================

def load_predictions(model_name):

    path = (
        REPORT_DIR
        / f"{model_name}_test_predictions.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        index_col="sample_id",
    ).sort_index()

    required = {
        "label",
        "score",
        "prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{model_name}: Spalten fehlen."
        )

    if len(df) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{model_name}: Falsche Sample-Anzahl."
        )

    if df.index.has_duplicates:
        raise ValueError(
            f"{model_name}: Doppelte Sample-IDs."
        )

    if df[["label", "score", "prediction"]].isna().any().any():
        raise ValueError(
            f"{model_name}: Fehlende Werte."
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"{model_name}: Ungültige Labels."
        )

    if not df["prediction"].isin([0, 1]).all():
        raise ValueError(
            f"{model_name}: Ungültige Vorhersagen."
        )

    scores = df["score"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError(
            f"{model_name}: Ungültige Wahrscheinlichkeiten."
        )

    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(
            f"{model_name}: Scores ausserhalb 0 bis 1."
        )

    predicted = (scores >= 0.5).astype(int)

    if not np.array_equal(
        predicted,
        df["prediction"].to_numpy(dtype=int),
    ):
        raise ValueError(
            f"{model_name}: Vorhersagen stimmen "
            "nicht mit dem Schwellenwert 0.5 überein."
        )

    return df


# ==================================================
# 5. KENNZAHLEN ABGLEICHEN
# ==================================================

def verify_metrics(
    model_name,
    predictions,
    saved,
):

    calculated = evaluate_predictions(
        model_name,
        predictions["label"].to_numpy(dtype=int),
        predictions["score"].to_numpy(dtype=float),
    )

    for key, calculated_value in calculated.items():

        if key == "model":
            continue

        if key not in saved.index:
            raise ValueError(
                f"{model_name}: Kennzahl {key} fehlt."
            )

        saved_value = float(saved[key])

        if not np.isclose(
            calculated_value,
            saved_value,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError(
                f"{model_name}: Kennzahl {key} "
                "stimmt nicht mit den Vorhersagen überein."
            )

    return calculated


# ==================================================
# 6. HAUPTAUSWERTUNG
# ==================================================

def main():

    print("=" * 65)
    print("WILDFIREFORECAST - FINALE MODELLERGEBNISSE")
    print("=" * 65)

    results = []

    reference_ids = None
    reference_labels = None

    for model_name, settings in MODELS.items():

        print(
            f"\nPrüfe Modell {model_name} ...",
            flush=True,
        )

        predictions = load_predictions(model_name)

        # Identische Samples und Labels sicherstellen
        if reference_ids is None:

            reference_ids = predictions.index.copy()

            reference_labels = (
                predictions["label"]
                .to_numpy(dtype=int)
                .copy()
            )

        else:

            if not predictions.index.equals(reference_ids):
                raise ValueError(
                    f"{model_name}: Sample-IDs stimmen "
                    "nicht mit den anderen Modellen überein."
                )

            if not np.array_equal(
                predictions["label"].to_numpy(dtype=int),
                reference_labels,
            ):
                raise ValueError(
                    f"{model_name}: Labels unterscheiden sich."
                )

        saved = load_saved_metrics(
            model_name,
            settings,
        )

        metrics = verify_metrics(
            model_name,
            predictions,
            saved,
        )

        results.append(
            {
                "model": model_name,
                "family": settings["family"],
                "version": settings["version"],
                "features": settings["features"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "model"
                },
            }
        )

        print("775 Samples: OK")
        print("Sample-Zuordnung: OK")
        print("Kennzahlen: OK")

    # ==================================================
    # 7. GEMEINSAME VERGLEICHSTABELLE
    # ==================================================

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        ["roc_auc", "model"],
        ascending=[False, True],
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank_roc_auc",
        np.arange(1, len(comparison) + 1),
    )

    display_columns = [
        "rank_roc_auc",
        "model",
        "family",
        "version",
        "features",
        "samples",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    comparison = comparison[display_columns]

    # Erst nach erfolgreicher Kontrolle aller
    # acht Modelle wird die Tabelle gespeichert.
    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ==================================================
    # 8. ABSCHLUSSBERICHT
    # ==================================================

    print("\n" + "=" * 65)
    print("ALLE ACHT MODELLE ERFOLGREICH KONTROLLIERT")
    print("=" * 65)

    print(
        comparison[
            [
                "rank_roc_auc",
                "model",
                "family",
                "accuracy",
                "f1",
                "roc_auc",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nAnzahl Test-Samples:", EXPECTED_SAMPLES)

    print("\nHöchste ROC-AUC:")
    best_auc = comparison.iloc[0]

    print(
        best_auc["model"],
        "-",
        round(best_auc["roc_auc"], 4),
    )

    print("\nHöchster F1-Score:")
    best_f1 = comparison.loc[
        comparison["f1"].idxmax()
    ]

    print(
        best_f1["model"],
        "-",
        round(best_f1["f1"], 4),
    )

    print("\nErgebnistabelle gespeichert:")
    print(OUTPUT_FILE)

    print("\nKeine Modelle trainiert oder getestet.")
    print("Nur bereits gespeicherte Ergebnisse ausgewertet.")


# ==================================================
# 9. PROGRAMMSTART
# ==================================================

if __name__ == "__main__":
    main()
