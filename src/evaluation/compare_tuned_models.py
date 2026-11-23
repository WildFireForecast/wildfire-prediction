
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# ==========================================
# 1. PROJEKTPFADE
# ==========================================

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "model_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = OUTPUT / "baseline_model_comparison.csv"

baseline = pd.read_csv(BASELINE_FILE)

baseline = baseline[
    [
        "model", "accuracy", "precision",
        "recall", "f1", "roc_auc",
        "average_precision"
    ]
].copy()

baseline["version"] = "baseline"

results = [baseline]

# ==========================================
# 2. XGBOOST UND TRANSFORMER - TUNED
# ==========================================

tuned_files = {
    "2A": REPORTS / "xgboost/tuning/2A_tuned_validation_predictions.csv",
    "2B": REPORTS / "xgboost/tuning/2B_tuned_validation_predictions.csv",
    "4A": REPORTS / "transformer/tuning/4A_best_validation_predictions.csv",
    "4B": REPORTS / "transformer/tuning/4B_best_validation_predictions.csv",
}

# Referenz zur Pruefung gleicher Samples
reference = pd.read_csv(
    REPORTS / "logistic_regression/baseline/1A_validation_predictions.csv",
    index_col=0,
).sort_index()

tuned_results = []

for name, path in tuned_files.items():

    df = pd.read_csv(path, index_col=0).sort_index()

    required = {"label", "score", "prediction"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{name}: Erwartete Spalten fehlen: "
            f"{required - set(df.columns)}"
        )

    if df.index.has_duplicates:
        raise ValueError(f"{name}: Doppelte Sample-IDs.")

    if not df.index.equals(reference.index):
        raise ValueError(f"{name}: Andere Sample-IDs.")

    if not df["label"].equals(reference["label"]):
        raise ValueError(f"{name}: Labels stimmen nicht.")

    y = df["label"].to_numpy(dtype=int)
    pred = df["prediction"].to_numpy(dtype=int)
    score = df["score"].to_numpy(dtype=float)

    if not np.isfinite(score).all():
        raise ValueError(f"{name}: Ungueltige Scores.")

    tuned_results.append({
        "model": name,
        "version": "tuned",
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(
            y, score
        ),
    })

results.append(pd.DataFrame(tuned_results))

# ==========================================
# 3. LSTM - TUNING-ZUSAMMENFASSUNG
# ==========================================

lstm = pd.read_csv(
    REPORTS / "lstm/tuning/tuning_summary.csv"
)

required = {"model", "roc_auc", "accuracy", "f1"}

if not required.issubset(lstm.columns):
    raise ValueError("LSTM-Tuning-Spalten fehlen.")

# Beste Konfiguration je Modell nach ROC-AUC
lstm_best = (
    lstm.sort_values("roc_auc", ascending=False)
    .drop_duplicates(subset="model")
    .copy()
)

lstm_best["version"] = "tuned"

# Diese Kennzahlen sind nicht in der Datei vorhanden
lstm_best["precision"] = np.nan
lstm_best["recall"] = np.nan
lstm_best["average_precision"] = np.nan

results.append(
    lstm_best[
        [
            "model", "version", "accuracy",
            "precision", "recall", "f1",
            "roc_auc", "average_precision"
        ]
    ]
)

# ==========================================
# 4. GESAMTVERGLEICH
# ==========================================

comparison = pd.concat(
    results, ignore_index=True
)

comparison = comparison.sort_values(
    "roc_auc", ascending=False
)

comparison.to_csv(
    OUTPUT / "all_model_comparison.csv",
    index=False,
)

print("\n===================================")
print(" GESAMTVERGLEICH - VALIDIERUNG 2020")
print("===================================")

print(
    comparison[
        ["model", "version", "accuracy", "f1", "roc_auc"]
    ].round(4).to_string(index=False)
)

# ==========================================
# 5. VERBESSERUNG DURCH TUNING
# ==========================================

base_auc = baseline.set_index("model")["roc_auc"]

tuned_auc = comparison.loc[
    comparison["version"].eq("tuned"),
    ["model", "roc_auc"]
].set_index("model")["roc_auc"]

improvement = pd.DataFrame({
    "baseline_auc": base_auc,
    "tuned_auc": tuned_auc
}).dropna()

improvement["difference"] = (
    improvement["tuned_auc"]
    - improvement["baseline_auc"]
)

improvement.to_csv(
    OUTPUT / "tuning_improvements.csv"
)

print("\n===================================")
print(" VERBESSERUNG DURCH TUNING")
print("===================================")
print(improvement.round(4).to_string())

# ==========================================
# 6. GRAFIK ERSTELLEN
# ==========================================

fig, ax = plt.subplots(figsize=(10, 6))

models = improvement.index.tolist()
positions = np.arange(len(models))
width = 0.35

ax.bar(
    positions - width / 2,
    improvement["baseline_auc"],
    width,
    label="Baseline"
)

ax.bar(
    positions + width / 2,
    improvement["tuned_auc"],
    width,
    label="Tuned"
)

ax.set_xticks(positions)
ax.set_xticklabels(models)
ax.set_ylim(0.85, 1.0)
ax.set_xlabel("Modellvariante")
ax.set_ylabel("ROC-AUC")
ax.set_title("Baseline vs. Tuning - Validierung 2020")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT / "baseline_vs_tuned_auc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nGesamtvergleich gespeichert unter:")
print(OUTPUT)
print("Testdaten wurden nicht verwendet.")
