
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ==========================================
# 1. Projektpfade
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data" / "processed" / "provisional"
BASELINE = DATA / "baseline"
OUTPUT = DATA / "xgboost_2B"

OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Konfiguration laden
# ==========================================

config = json.loads(
    (DATA / "input_config.json").read_text(
        encoding="utf-8"
    )
)

original_features = config["features_with_lst"]

# Niederschlagsgrenze für niederschlagsarme Tage:
# 1 mm = 0.001 m
# Vorläufige, dokumentierte Modellentscheidung.

DRY_THRESHOLD = 0.001

# ==========================================
# 3. Zusätzliche Features definieren
# ==========================================

new_features = {
    # Niederschlagssummen
    "tp_sum_7d": ("tp", "sum", 7),
    "tp_sum_14d": ("tp", "sum", 14),
    "tp_sum_30d": ("tp", "sum", 30),

    # Temperatur
    "t2m_mean_7d": ("t2m", "mean", 7),
    "t2m_max_7d": ("t2m", "max", 7),
    "t2m_mean_30d": ("t2m", "mean", 30),
    "t2m_max_30d": ("t2m", "max", 30),

    # Bodenfeuchtigkeit
    "smi_mean_7d": ("smi", "mean", 7),
    "smi_mean_30d": ("smi", "mean", 30),

    # Relative Luftfeuchtigkeit
    "rh_mean_7d": ("rh", "mean", 7),
    "rh_mean_30d": ("rh", "mean", 30),

    # Windgeschwindigkeit
    "wind_speed_mean_7d": ("wind_speed", "mean", 7),

    # Trockenperioden
    "dry_days_30d": ("tp", "dry_days", 30),
    "dry_streak_59d": ("tp", "dry_streak", 59),
}

# ==========================================
# 4. Funktion zur Feature-Berechnung
# ==========================================

def calculate_feature(group, column, operation, days):

    # Letzte n Tage des Samples
    values = group[column].tail(days)

    # Unvollständige Fenster oder fehlende Werte
    if len(values) != days or values.isna().any():
        return np.nan

    # Negative Niederschlagswerte auf null setzen
    # Nur für die neuen Features
    if column == "tp":
        values = values.clip(lower=0)

    # Summe
    if operation == "sum":
        return float(values.sum())

    # Durchschnitt
    elif operation == "mean":
        return float(values.mean())

    # Maximum
    elif operation == "max":
        return float(values.max())

    # Anzahl niederschlagsarmer Tage
    elif operation == "dry_days":
        return int(
            (values < DRY_THRESHOLD).sum()
        )

    # Zusammenhängende Trockenperiode am Ende
    elif operation == "dry_streak":

        dry = values < DRY_THRESHOLD
        streak = 0

        # Rückwärts zählen, beginnend bei Tag 58
        for is_dry in reversed(dry.to_numpy()):

            if is_dry:
                streak += 1

            else:
                break

        return int(streak)

    else:
        raise ValueError(
            f"Unbekannte Operation: {operation}"
        )


# ==========================================
# 5. Features für Train und Validation
# ==========================================

for split in ["train", "validation"]:

    print(f"\n=== VERARBEITE {split.upper()} ===")

    # Zeitreihendaten laden
    df = pd.read_csv(
        DATA / f"{split}_sequences.csv"
    )

    df = df.sort_values(
        ["sample_id", "time_idx"]
    )

    # ======================================
    # 5.1 Zeitreihen überprüfen
    # ======================================

    valid = df.groupby("sample_id")["time_idx"].agg(
        lambda x: len(x) == 59
        and set(x) == set(range(59))
    )

    if not valid.all():
        raise ValueError(
            f"{split}: Unvollständige Zeitreihen."
        )

    if df.duplicated(
        ["sample_id", "time_idx"]
    ).any():
        raise ValueError(
            f"{split}: Doppelte Zeitindizes."
        )

    # ======================================
    # 5.2 Baseline von Modell 2A laden
    # ======================================

    baseline = pd.read_csv(
        BASELINE / f"{split}_X_with_lst.csv",
        index_col="sample_id"
    ).sort_index()

    if not baseline.index.is_unique:
        raise ValueError(
            f"{split}: Doppelte Baseline-Samples."
        )

    if list(baseline.columns) != original_features:
        raise ValueError(
            f"{split}: Unerwartete Baseline-Features."
        )

    # ======================================
    # 5.3 Neue Features berechnen
    # ======================================

    records = []

    for sample_id, group in df.groupby("sample_id"):

        group = group.sort_values("time_idx")

        record = {
            "sample_id": sample_id
        }

        for name, (
            column,
            operation,
            days
        ) in new_features.items():

            value = calculate_feature(
                group,
                column,
                operation,
                days
            )

            record[name] = value

        records.append(record)

    engineered = (
        pd.DataFrame(records)
        .set_index("sample_id")
        .sort_index()
    )

    # ======================================
    # 5.4 Sample-IDs kontrollieren
    # ======================================

    if set(engineered.index) != set(baseline.index):
        raise ValueError(
            f"{split}: Sample-IDs stimmen nicht überein."
        )

    if not engineered.index.is_unique:
        raise ValueError(
            f"{split}: Doppelte Samples."
        )

    # ======================================
    # 5.5 Originalfeatures + neue Features
    # ======================================

    X = baseline.join(
        engineered,
        validate="one_to_one"
    )

    expected_features = (
        original_features
        + list(new_features.keys())
    )

    if list(X.columns) != expected_features:
        raise ValueError(
            f"{split}: Unerwartete Feature-Liste."
        )

    if len(X) != len(baseline):
        raise ValueError(
            f"{split}: Anzahl Samples verändert."
        )

    # ======================================
    # 5.6 Neue Eingabedaten speichern
    # ======================================

    X.to_csv(
        OUTPUT / f"{split}_X.csv"
    )

    # ======================================
    # 5.7 Ausgabe und Kontrolle
    # ======================================

    print("Samples:", len(X))
    print("Originalfeatures:", len(original_features))
    print("Neue Features:", len(new_features))
    print("Features insgesamt:", X.shape[1])

    print("\nFehlende Werte in neuen Features:")
    print(
        engineered.isna().sum().to_string()
    )

    print("\nTrockenperioden (Zusammenfassung):")
    print(
        engineered[
            ["dry_days_30d", "dry_streak_59d"]
        ].describe().round(2).to_string()
    )

# ==========================================
# 6. Feature-Konfiguration speichern
# ==========================================

feature_config = {
    "model": "2B",
    "status": "provisional",
    "algorithm": "XGBoost",
    "feature_engineering": True,

    "original_features": original_features,
    "engineered_features": new_features,

    "total_features": (
        len(original_features)
        + len(new_features)
    ),

    "input_indices": [0, 58],
    "target_day_assumption": config["target_day_assumption"],

    "precipitation_unit": "m",

    "dry_spell_features": [
        "dry_days_30d",
        "dry_streak_59d",
    ],

    "dry_threshold_m": DRY_THRESHOLD,
    "dry_threshold_mm": 1.0,

    "negative_tp_handling": (
        "Clip negative precipitation values to zero "
        "for engineered features only"
    ),

    "missing_value_handling": (
        "Incomplete windows become NaN"
    ),

    "training_years": "2006-2019",
    "validation_year": 2020,
    "test_used": False,
}

(OUTPUT / "feature_config.json").write_text(
    json.dumps(
        feature_config,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

# ==========================================
# 7. Abschluss
# ==========================================

print("\n====================================")
print("FEATURE ENGINEERING 2B ABGESCHLOSSEN")
print("====================================")

print("Originalfeatures:", len(original_features))
print("Neue Features:", len(new_features))
print(
    "Features insgesamt:",
    len(original_features) + len(new_features)
)

print("\nGespeichert unter:")
print(OUTPUT)

print("\nTestdaten wurden nicht verwendet.")
