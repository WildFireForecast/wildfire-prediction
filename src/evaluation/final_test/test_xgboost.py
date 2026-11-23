
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

DATA = ROOT / "data/processed/provisional"
BASELINE = DATA / "baseline"
FEATURE_2B = DATA / "xgboost_2B"

MODELS_DIR = ROOT / "models/xgboost/tuned"
REPORTS = ROOT / "reports/final_test"

SEQ_LEN = 59
DRY_THRESHOLD = 0.001


# ==================================================
# 2. MODELLKONFIGURATION
# ==================================================

MODELS = {
    "2A": {
        "model": MODELS_DIR / "xgboost_2A_tuned.joblib",
        "train": BASELINE / "train_X_with_lst.csv",
        "test": BASELINE / "test_X_with_lst.csv",
        "features": 27,
    },
    "2B": {
        "model": MODELS_DIR / "xgboost_2B_tuned.joblib",
        "train": FEATURE_2B / "train_X.csv",
        "test": FEATURE_2B / "test_X.csv",
        "features": 41,
    },
}


# ==================================================
# 3. FEATURE ENGINEERING 2B
# IDENTISCH MIT DEM ORIGINAL
# ==================================================

NEW_FEATURES = {
    "tp_sum_7d": ("tp", "sum", 7),
    "tp_sum_14d": ("tp", "sum", 14),
    "tp_sum_30d": ("tp", "sum", 30),

    "t2m_mean_7d": ("t2m", "mean", 7),
    "t2m_max_7d": ("t2m", "max", 7),
    "t2m_mean_30d": ("t2m", "mean", 30),
    "t2m_max_30d": ("t2m", "max", 30),

    "smi_mean_7d": ("smi", "mean", 7),
    "smi_mean_30d": ("smi", "mean", 30),

    "rh_mean_7d": ("rh", "mean", 7),
    "rh_mean_30d": ("rh", "mean", 30),

    "wind_speed_mean_7d": (
        "wind_speed", "mean", 7
    ),

    "dry_days_30d": ("tp", "dry_days", 30),
    "dry_streak_59d": ("tp", "dry_streak", 59),
}


# ==================================================
# 4. ORIGINAL-FEATURE-BERECHNUNG
# ==================================================

def calculate_feature(group, column, operation, days):

    values = group[column].tail(days)

    if len(values) != days or values.isna().any():
        return np.nan

    if column == "tp":
        values = values.clip(lower=0)

    if operation == "sum":
        return float(values.sum())

    if operation == "mean":
        return float(values.mean())

    if operation == "max":
        return float(values.max())

    if operation == "dry_days":
        return int((values < DRY_THRESHOLD).sum())

    if operation == "dry_streak":
        dry = values < DRY_THRESHOLD
        streak = 0

        for is_dry in reversed(dry.to_numpy()):
            if is_dry:
                streak += 1
            else:
                break

        return int(streak)

    raise ValueError(
        f"Unbekannte Feature-Operation: {operation}"
    )


# ==================================================
# 5. MODELL UND FEATURE-REIHENFOLGE PRÜFEN
# ==================================================

def get_features(settings):

    columns = pd.read_csv(
        settings["train"],
        nrows=0,
        index_col="sample_id",
    ).columns.tolist()

    if len(columns) != settings["features"]:
        raise ValueError(
            "Feature-Anzahl weicht vom Training ab."
        )

    return columns


def load_checked_model(name):

    settings = MODELS[name]

    model = joblib.load(settings["model"])
    features = get_features(settings)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{name}: predict_proba fehlt."
        )

    if model.n_features_in_ != len(features):
        raise ValueError(
            f"{name}: Feature-Anzahl stimmt nicht."
        )

    booster_features = model.get_booster().feature_names

    if booster_features is not None:
        if list(booster_features) != features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )

    return model, features


# ==================================================
# 6. SICHERE MODELLPRÜFUNG
# KEINE TESTDATEN
# ==================================================

def check_models():

    print("=" * 60)
    print("XGBOOST - MODELLPRÜFUNG")
    print("=" * 60)

    for name in MODELS:

        print(f"\nPrüfe Modell {name} ...")

        model, features = load_checked_model(name)

        print("Modell geladen: OK")
        print("Feature-Reihenfolge: OK")
        print("Features:", len(features))
        print("Vorhersagefunktion vorhanden: OK")

    print("\nBeide XGBoost-Modelle erfolgreich geprüft.")
    print("Keine Testdaten verwendet.")


# ==================================================
# 7. TESTFEATURES FÜR 2B VORBEREITEN
# ==================================================

def prepare_test_features_2B():

    require_final_test_approval()

    print("\nPrüfe XGBoost 2B Testfeatures ...")

    config = json.loads(
        (DATA / "input_config.json").read_text(
            encoding="utf-8"
        )
    )

    original_features = config["features_with_lst"]

    expected_features = (
        original_features + list(NEW_FEATURES)
    )

    # Bestehende Trainingskonfiguration prüfen
    feature_config = json.loads(
        (FEATURE_2B / "feature_config.json").read_text(
            encoding="utf-8"
        )
    )

    if feature_config["original_features"] != original_features:
        raise ValueError(
            "Originalfeatures stimmen nicht überein."
        )

    saved_features = {
        name: tuple(values)
        for name, values in feature_config[
            "engineered_features"
        ].items()
    }

    if saved_features != NEW_FEATURES:
        raise ValueError(
            "Feature Engineering weicht vom Training ab."
        )

    if not np.isclose(
        feature_config["dry_threshold_m"],
        DRY_THRESHOLD,
    ):
        raise ValueError(
            "Trockenheitsgrenze stimmt nicht überein."
        )

    training_features = get_features(MODELS["2B"])

    if training_features != expected_features:
        raise ValueError(
            "Feature-Reihenfolge weicht vom Training ab."
        )

    output_path = MODELS["2B"]["test"]

    # Eine vorhandene Datei nicht überschreiben
    if output_path.exists():

        X = pd.read_csv(
            output_path,
            index_col="sample_id",
        ).sort_index()

        validate_test_features_2B(
            X, expected_features
        )

        print("Vorhandene Testfeatures geprüft: OK")

        return X

    # Testsequenzen laden
    sequences = pd.read_csv(
        DATA / "test_sequences.csv"
    )

    if sequences.duplicated(
        ["sample_id", "time_idx"]
    ).any():
        raise ValueError(
            "Doppelte Zeitindizes gefunden."
        )

    sequences = sequences.sort_values(
        ["sample_id", "time_idx"]
    )

    valid = sequences.groupby(
        "sample_id"
    )["time_idx"].agg(
        lambda values: np.array_equal(
            values.to_numpy(),
            np.arange(SEQ_LEN),
        )
    )

    if not valid.all():
        raise ValueError(
            "Unvollständige Testsequenzen."
        )

    # 27 bestehende Basisfeatures laden
    baseline = pd.read_csv(
        BASELINE / "test_X_with_lst.csv",
        index_col="sample_id",
    ).sort_index()

    if baseline.index.has_duplicates:
        raise ValueError(
            "Doppelte Test-Sample-IDs."
        )

    if list(baseline.columns) != original_features:
        raise ValueError(
            "Baseline-Features stimmen nicht."
        )

    if set(sequences["sample_id"]) != set(baseline.index):
        raise ValueError(
            "Sample-IDs der Sequenzen stimmen nicht."
        )

    # 14 zusätzliche Features berechnen
    records = []

    for sample_id, group in sequences.groupby(
        "sample_id"
    ):

        record = {"sample_id": sample_id}

        for name, (
            column,
            operation,
            days,
        ) in NEW_FEATURES.items():

            record[name] = calculate_feature(
                group,
                column,
                operation,
                days,
            )

        records.append(record)

    engineered = (
        pd.DataFrame(records)
        .set_index("sample_id")
        .sort_index()
    )

    if not engineered.index.equals(baseline.index):
        raise ValueError(
            "Sample-IDs stimmen nicht überein."
        )

    # 27 + 14 = 41 Features
    X = baseline.join(
        engineered,
        validate="one_to_one",
    )

    validate_test_features_2B(
        X, expected_features
    )

    FEATURE_2B.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        raise FileExistsError(
            "Testdatei existiert bereits."
        )

    X.to_csv(output_path)

    print("Testfeatures erstellt: OK")
    print("Samples:", len(X))
    print("Features:", X.shape[1])
    print("Datei:", output_path)

    return X


# ==================================================
# 8. TESTFEATURES VALIDIEREN
# ==================================================

def validate_test_features_2B(X, expected_features):

    if X.empty or X.index.has_duplicates:
        raise ValueError(
            "Ungültige Test-Sample-IDs."
        )

    if list(X.columns) != expected_features:
        raise ValueError(
            "Falsche Feature-Reihenfolge."
        )

    if X.shape != (775, 41):
        raise ValueError(
            "Erwartet werden 775 Samples mit 41 Features."
        )

    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(
            "Unendliche Feature-Werte gefunden."
        )


# ==================================================
# 9. FINALE TESTEVALUATION
# ==================================================

def run_final_evaluation():

    # Sperre VOR jeglichem Testdatenzugriff
    require_final_test_approval()

    if "--execute" not in sys.argv:
        raise RuntimeError(
            "Für den Test ist --execute erforderlich."
        )

    print("=" * 60)
    print("XGBOOST - FINALE TESTEVALUATION")
    print("=" * 60)

    # Fehlende 2B-Testfeatures erstellen
    X_2B = prepare_test_features_2B()

    y = pd.read_csv(
        BASELINE / "test_y.csv",
        index_col="sample_id",
    )["label"].sort_index()

    if y.empty or y.index.has_duplicates:
        raise ValueError(
            "Ungültige Testlabels."
        )

    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError(
            "Testlabels müssen beide Klassen enthalten."
        )

    if len(y) != 775:
        raise ValueError(
            "Unerwartete Anzahl Testlabels."
        )

    results = []
    prediction_files = {}

    for name, settings in MODELS.items():

        print(f"\nEvaluiere Modell {name} ...")

        model, features = load_checked_model(name)

        if name == "2B":
            X = X_2B
        else:
            X = pd.read_csv(
                settings["test"],
                index_col="sample_id",
            ).sort_index()

        if X.index.has_duplicates:
            raise ValueError(
                f"{name}: Doppelte Sample-IDs."
            )

        if not X.index.equals(y.index):
            raise ValueError(
                f"{name}: Sample-IDs stimmen nicht überein."
            )

        if list(X.columns) != features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )

        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError(
                f"{name}: Unendliche Feature-Werte."
            )

        # Gespeichertes Modell: keine neue Anpassung
        scores = model.predict_proba(X)[:, 1]

        metrics = evaluate_predictions(
            name,
            y.to_numpy(),
            scores,
        )

        metrics["version"] = "tuned"
        results.append(metrics)

        predictions = (scores >= 0.5).astype(int)

        prediction_df = pd.DataFrame(
            {
                "label": y.to_numpy(),
                "score": scores,
                "prediction": predictions,
            },
            index=y.index,
        )

        prediction_df.index.name = "sample_id"
        prediction_files[name] = prediction_df

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

    # Ergebnisse speichern
    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in prediction_files.items():
        df.to_csv(
            REPORTS / f"{name}_test_predictions.csv"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORTS / "xgboost_test_metrics.csv",
        index=False,
    )

    print("\nXGBOOST - TEST ABGESCHLOSSEN")
    print(results_df.to_string(index=False))
    print("\nErgebnisse:", REPORTS)


# ==================================================
# 10. PROGRAMMSTART
# ==================================================

def main():

    if "--execute" not in sys.argv:
        check_models()
        return

    run_final_evaluation()


if __name__ == "__main__":
    main()
