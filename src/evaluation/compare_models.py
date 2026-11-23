
from pathlib import Path
import pandas as pd
import numpy as np

# Projektverzeichnis
ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "model_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Gespeicherte Vorhersagen der Modellvarianten
MODEL_FILES = {
    "1A": REPORTS / "logistic_regression/baseline/1A_validation_predictions.csv",
    "1B": REPORTS / "logistic_regression/baseline/1B_validation_predictions.csv",
    "2A": REPORTS / "xgboost/baseline_2A/2A_validation_predictions.csv",
    "2B": REPORTS / "xgboost/baseline_2B/2B_validation_predictions.csv",
    "3A": REPORTS / "lstm/baseline/3A_validation_predictions.csv",
    "3B": REPORTS / "lstm/baseline/3B_validation_predictions.csv",
    "4A": REPORTS / "transformer/baseline/4A_validation_predictions.csv",
    "4B": REPORTS / "transformer/baseline/4B_validation_predictions.csv",
}

reference_ids = None
reference_labels = None

print("\n========================================")
print(" PRUEFUNG DER MODELLVERGLEICHBARKEIT")
print("========================================")

for name, path in MODEL_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Fehlende Ergebnisdatei fuer Modell {name}: {path}"
        )

    df = pd.read_csv(path, index_col=0)

    if not {"label", "score", "prediction"}.issubset(df.columns):
        raise ValueError(
            f"Modell {name}: Erforderliche Spalten fehlen."
        )

    if df.index.has_duplicates:
        raise ValueError(
            f"Modell {name}: Doppelte Sample-IDs."
        )

    df = df.sort_index()

    if df[["label", "score", "prediction"]].isna().any().any():
        raise ValueError(
            f"Modell {name}: Fehlende Werte."
        )

    if not np.isfinite(df["score"].to_numpy()).all():
        raise ValueError(
            f"Modell {name}: Ungueltige Scores."
        )

    if not df["score"].between(0, 1).all():
        raise ValueError(
            f"Modell {name}: Scores liegen ausserhalb 0 bis 1."
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"Modell {name}: Ungueltige Labels."
        )

    if not df["prediction"].isin([0, 1]).all():
        raise ValueError(
            f"Modell {name}: Ungueltige Vorhersagen."
        )

    if reference_ids is None:
        reference_ids = df.index
        reference_labels = df["label"].copy()
    else:
        if not df.index.equals(reference_ids):
            raise ValueError(
                f"Modell {name}: Sample-IDs stimmen nicht ueberein."
            )

        if not df["label"].equals(reference_labels):
            raise ValueError(
                f"Modell {name}: Labels stimmen nicht ueberein."
            )

    print(
        f"Modell {name}: {len(df)} Samples - OK"
    )

print("\nAlle acht Modelle verwenden identische")
print("Sample-IDs und Zielwerte.")
print("Vergleichbarkeitspruefung erfolgreich!")
print("Testdaten wurden nicht verwendet.")


# ==================================================
# SCHRITT 3: MODELLVERGLEICH BERECHNEN
# ==================================================

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

results = []

for name, path in MODEL_FILES.items():

    df = pd.read_csv(path, index_col=0)
    df = df.sort_index()

    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df["prediction"].to_numpy(dtype=int)
    y_score = df["score"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(
            y_true, y_score
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    })

comparison = pd.DataFrame(results)

# Nach ROC-AUC absteigend sortieren
comparison = comparison.sort_values(
    "roc_auc", ascending=False
)

# Ergebnisse speichern
comparison.to_csv(
    OUTPUT / "baseline_model_comparison.csv",
    index=False,
)

print("\n========================================")
print(" MODELLVERGLEICH - VALIDIERUNG 2020")
print("========================================")

columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]

print(
    comparison[columns]
    .round(4)
    .to_string(index=False)
)

print("\nVergleichstabelle gespeichert unter:")
print(OUTPUT / "baseline_model_comparison.csv")
print("Testdaten wurden nicht verwendet.")


# ==================================================
# SCHRITT 4: ROC- UND PRECISION-RECALL-KURVEN
# ==================================================

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
)

# Einheitliche Modellbezeichnungen
MODEL_LABELS = {
    "1A": "1A - Logistic Regression (ohne LST)",
    "1B": "1B - Logistic Regression (mit LST)",
    "2A": "2A - XGBoost (ohne FE)",
    "2B": "2B - XGBoost (mit FE)",
    "3A": "3A - LSTM (ohne LST)",
    "3B": "3B - LSTM (mit LST)",
    "4A": "4A - Transformer (ohne LST)",
    "4B": "4B - Transformer (mit LST)",
}

# -------------------------------
# ROC-Kurven
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

for name, path in MODEL_FILES.items():

    df = pd.read_csv(path, index_col=0)
    df = df.sort_index()

    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{MODEL_LABELS[name]} (AUC={roc_auc:.3f})"
    )

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Zufallsklassifikator"
)

ax.set_title("ROC-Kurven – Validierung 2020")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT / "roc_curves_baseline.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("ROC-Kurven gespeichert.")


# -------------------------------
# Precision-Recall-Kurven
# -------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

for name, path in MODEL_FILES.items():

    df = pd.read_csv(path, index_col=0)
    df = df.sort_index()

    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()

    precision, recall, _ = precision_recall_curve(
        y_true, y_score
    )

    ap = average_precision_score(y_true, y_score)

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{MODEL_LABELS[name]} (AP={ap:.3f})"
    )

# Referenz: Anteil positiver Beispiele
positive_rate = reference_labels.mean()

ax.axhline(
    positive_rate,
    linestyle="--",
    color="gray",
    label=f"Positiver Klassenanteil ({positive_rate:.3f})"
)

ax.set_title("Precision-Recall-Kurven – Validierung 2020")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT / "precision_recall_curves_baseline.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Precision-Recall-Kurven gespeichert.")
print("Alle Grafiken basieren auf der Validierung 2020.")
