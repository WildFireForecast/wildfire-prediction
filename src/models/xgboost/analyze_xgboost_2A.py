
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

MODEL_FILE = ROOT / "models/provisional/xgboost_2A.joblib"
CONFIG_FILE = (
    ROOT / "data/processed/provisional/input_config.json"
)
REPORTS = ROOT / "reports/xgboost_provisional"

REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Modell und Features laden
# ==========================================

model = joblib.load(MODEL_FILE)

config = json.loads(
    CONFIG_FILE.read_text(encoding="utf-8")
)

features = config["features_with_lst"]

booster = model.get_booster()

if booster.feature_names is not None:
    if booster.feature_names != features:
        raise ValueError(
            "Feature-Namen stimmen nicht mit dem Modell überein."
        )

# ==========================================
# 3. Feature Importance berechnen
# ==========================================

# Gain misst die durchschnittliche Verbesserung
# durch Splits, die das jeweilige Merkmal nutzen.

raw_importance = booster.get_score(
    importance_type="gain"
)

importance = []

for i, feature in enumerate(features):
    # XGBoost kann Feature-Namen oder f0, f1 usw. nutzen.
    gain = raw_importance.get(
        feature,
        raw_importance.get(f"f{i}", 0.0)
    )

    importance.append({
        "feature": feature,
        "importance_gain": float(gain),
    })

df = pd.DataFrame(importance)

df = df.sort_values(
    "importance_gain",
    ascending=False
).reset_index(drop=True)

total = df["importance_gain"].sum()

if total <= 0:
    raise ValueError("Keine positiven Importance-Werte.")

df["importance_percent"] = (
    df["importance_gain"] / total * 100
)

df["rank"] = range(1, len(df) + 1)

# ==========================================
# 4. Ergebnisse speichern
# ==========================================

df.to_csv(
    REPORTS / "2A_feature_importance.csv",
    index=False
)

print("\n=== FEATURE IMPORTANCE MODELL 2A ===")
print(
    df[
        ["rank", "feature", "importance_percent"]
    ].round(2).to_string(index=False)
)

# ==========================================
# 5. Balkendiagramm erstellen
# ==========================================

top15 = df.head(15).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top15["feature"],
    top15["importance_percent"]
)

ax.set_xlabel("Relativer Gain-Anteil (%)")
ax.set_ylabel("Umweltmerkmal")

ax.set_title(
    "WildFireForecast – Modell 2A\n"
    "Top 15 Feature Importances (XGBoost Gain)"
)

ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    REPORTS / "2A_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAnalyse erfolgreich abgeschlossen.")
print("CSV und Diagramm gespeichert unter:")
print(REPORTS)
