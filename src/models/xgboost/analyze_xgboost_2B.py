
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. Projektpfade
# ==========================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = ROOT / "models/provisional/xgboost_2B.joblib"
CONFIG_FILE = (
    ROOT / "data/processed/provisional/xgboost_2B/feature_config.json"
)

REPORTS = ROOT / "reports/xgboost_2B_provisional"
REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Modell und Feature-Konfiguration laden
# ==========================================

model = joblib.load(MODEL_FILE)

config = json.loads(
    CONFIG_FILE.read_text(encoding="utf-8")
)

original_features = config["original_features"]
engineered_features = list(
    config["engineered_features"].keys()
)

features = original_features + engineered_features

if len(features) != 41:
    raise ValueError("Erwartet werden 41 Features.")

booster = model.get_booster()

if booster.feature_names is not None:
    if booster.feature_names != features:
        raise ValueError(
            "Feature-Namen stimmen nicht mit dem Modell überein."
        )

# ==========================================
# 3. Feature Importance berechnen
# ==========================================

raw_importance = booster.get_score(
    importance_type="gain"
)

records = []

for i, feature in enumerate(features):

    gain = raw_importance.get(
        feature,
        raw_importance.get(f"f{i}", 0.0)
    )

    records.append({
        "feature": feature,
        "importance_gain": float(gain),
        "feature_type": (
            "Engineered"
            if feature in engineered_features
            else "Original"
        ),
    })

df = pd.DataFrame(records)

total_gain = df["importance_gain"].sum()

if total_gain <= 0:
    raise ValueError("Keine positiven Importance-Werte.")

df["importance_percent"] = (
    df["importance_gain"] / total_gain * 100
)

df = df.sort_values(
    "importance_gain",
    ascending=False
).reset_index(drop=True)

df["rank"] = range(1, len(df) + 1)

# ==========================================
# 4. Rangliste speichern
# ==========================================

df.to_csv(
    REPORTS / "2B_feature_importance.csv",
    index=False
)

print("\n=== FEATURE IMPORTANCE MODELL 2B ===")

print(
    df[
        ["rank", "feature", "feature_type", "importance_percent"]
    ].round(2).to_string(index=False)
)

# ==========================================
# 5. Neue Features separat untersuchen
# ==========================================

engineered = df[
    df["feature_type"] == "Engineered"
].copy()

print("\n=== NEUE FEATURES (2B) ===")

print(
    engineered[
        ["rank", "feature", "importance_percent"]
    ].round(2).to_string(index=False)
)

print("\nSumme relativer Gain-Anteile:")

print(
    df.groupby("feature_type")["importance_percent"]
    .sum()
    .round(2)
    .to_string()
)

# ==========================================
# 6. Top-15-Diagramm erstellen
# ==========================================

top15 = df.head(15).iloc[::-1]

fig, ax = plt.subplots(figsize=(11, 7))

colors = [
    "#0D9488" if t == "Engineered" else "#94A3B8"
    for t in top15["feature_type"]
]

ax.barh(
    top15["feature"],
    top15["importance_percent"],
    color=colors,
)

ax.set_xlabel("Relativer Gain-Anteil (%)")
ax.set_ylabel("Merkmal")

ax.set_title(
    "WildFireForecast – Modell 2B\n"
    "Top 15 Feature Importances (XGBoost Gain)"
)

ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

from matplotlib.patches import Patch

ax.legend(
    handles=[
        Patch(color="#0D9488", label="Neues Feature"),
        Patch(color="#94A3B8", label="Originalfeature"),
    ]
)

plt.tight_layout()

plt.savefig(
    REPORTS / "2B_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ==========================================
# 7. Vergleich mit Modell 2A
# ==========================================

importance_2A = pd.read_csv(
    ROOT / "reports/xgboost_provisional/2A_feature_importance.csv"
)

comparison = df.merge(
    importance_2A[
        ["feature", "importance_percent"]
    ].rename(
        columns={"importance_percent": "importance_2A"}
    ),
    on="feature",
    how="left",
)

comparison = comparison.rename(
    columns={"importance_percent": "importance_2B"}
)

comparison["importance_2A"] = (
    comparison["importance_2A"].fillna(0)
)

comparison["difference_pp"] = (
    comparison["importance_2B"]
    - comparison["importance_2A"]
)

comparison.to_csv(
    REPORTS / "comparison_feature_importance_2A_2B.csv",
    index=False,
)

print("\n=== VERGLEICH 2A VS. 2B ===")

print(
    comparison[
        [
            "feature",
            "feature_type",
            "importance_2A",
            "importance_2B",
            "difference_pp",
        ]
    ].head(20).round(2).to_string(index=False)
)

print("\nAnalyse erfolgreich abgeschlossen.")
print("Ergebnisse gespeichert unter:", REPORTS)
print("Testdaten wurden nicht verwendet.")
