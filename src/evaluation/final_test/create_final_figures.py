
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


# ==================================================
# 1. PROJEKTPFADE
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = ROOT / "reports/final_test"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "1A", "1B",
    "2A", "2B",
    "3A", "3B",
    "4A", "4B",
]

MODEL_NAMES = {
    "1A": "LogReg 1A",
    "1B": "LogReg 1B",
    "2A": "XGBoost 2A",
    "2B": "XGBoost 2B",
    "3A": "LSTM 3A",
    "3B": "LSTM 3B",
    "4A": "Transformer 4A",
    "4B": "Transformer 4B",
}


# ==================================================
# 2. GESPEICHERTE ERGEBNISSE LADEN
# ==================================================

def load_data():

    comparison = pd.read_csv(
        REPORT_DIR / "final_test_comparison.csv"
    )

    if set(comparison["model"]) != set(MODELS):
        raise ValueError(
            "Die Vergleichstabelle enthält nicht "
            "genau die acht erwarteten Modelle."
        )

    predictions = {}
    reference_ids = None
    reference_labels = None

    for model in MODELS:

        path = (
            REPORT_DIR
            / f"{model}_test_predictions.csv"
        )

        df = pd.read_csv(
            path,
            index_col="sample_id",
        ).sort_index()

        if len(df) != 775:
            raise ValueError(
                f"{model}: Unerwartete Sample-Anzahl."
            )

        if df.index.has_duplicates:
            raise ValueError(
                f"{model}: Doppelte Sample-IDs."
            )

        if df[["label", "score", "prediction"]].isna().any().any():
            raise ValueError(
                f"{model}: Fehlende Werte."
            )

        if reference_ids is None:
            reference_ids = df.index.copy()
            reference_labels = df["label"].to_numpy()
        else:
            if not df.index.equals(reference_ids):
                raise ValueError(
                    f"{model}: Sample-IDs unterscheiden sich."
                )

            if not np.array_equal(
                df["label"].to_numpy(),
                reference_labels,
            ):
                raise ValueError(
                    f"{model}: Labels unterscheiden sich."
                )

        if not np.isfinite(df["score"].to_numpy()).all():
            raise ValueError(
                f"{model}: Ungültige Scores."
            )

        predictions[model] = df

    return comparison, predictions


# ==================================================
# 3. ROC-KURVEN
# ==================================================

def plot_roc(predictions):

    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODELS:

        df = predictions[model]

        y_true = df["label"].to_numpy()
        y_score = df["score"].to_numpy()

        fpr, tpr, _ = roc_curve(
            y_true, y_score
        )

        auc = roc_auc_score(
            y_true, y_score
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=1.8,
            label=f"{MODEL_NAMES[model]} (AUC={auc:.3f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="gray",
        label="Zufallsklassifikator",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC-Kurven – unabhängiger Test 2021–2022",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================================================
# 4. PRECISION-RECALL-KURVEN
# ==================================================

def plot_precision_recall(predictions):

    fig, ax = plt.subplots(figsize=(10, 7))

    for model in MODELS:

        df = predictions[model]

        y_true = df["label"].to_numpy()
        y_score = df["score"].to_numpy()

        precision, recall, _ = (
            precision_recall_curve(
                y_true, y_score
            )
        )

        ap = average_precision_score(
            y_true, y_score
        )

        ax.plot(
            recall,
            precision,
            linewidth=1.8,
            label=f"{MODEL_NAMES[model]} (AP={ap:.3f})",
        )

    prevalence = predictions["1A"]["label"].mean()

    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1,
        color="gray",
        label=f"Klassenhäufigkeit ({prevalence:.3f})",
    )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall-Kurven – Test 2021–2022",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )

    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================================================
# 5. VERGLEICH ROC-AUC UND F1-SCORE
# ==================================================

def plot_comparison(comparison):

    df = comparison.sort_values(
        "roc_auc",
        ascending=True,
    )

    labels = [
        MODEL_NAMES[model]
        for model in df["model"]
    ]

    positions = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 7))

    bars_auc = ax.barh(
        positions - width / 2,
        df["roc_auc"],
        height=width,
        label="ROC-AUC",
    )

    bars_f1 = ax.barh(
        positions + width / 2,
        df["f1"],
        height=width,
        label="F1-Score",
    )

    ax.set_yticks(positions, labels)

    ax.set(
        xlabel="Kennzahl (0–1)",
        title="Modellvergleich – unabhängiger Test 2021–2022",
        xlim=(0, 1.08),
    )

    ax.bar_label(
        bars_auc,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.bar_label(
        bars_f1,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

    ax.legend()
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================================================
# 6. CONFUSION MATRICES
# ==================================================

def plot_confusion_matrices(predictions):

    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 8),
        constrained_layout=True,
    )

    for ax, model in zip(axes.flat, MODELS):

        df = predictions[model]

        matrix = confusion_matrix(
            df["label"],
            df["prediction"],
            labels=[0, 1],
        )

        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=458,
        )

        for i in range(2):
            for j in range(2):

                value = matrix[i, j]

                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=(
                        "white"
                        if value > 229
                        else "black"
                    ),
                )

        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])

        ax.set(
            title=MODEL_NAMES[model],
            xlabel="Vorhergesagt",
            ylabel="Tatsächlich",
        )

    fig.suptitle(
        "Confusion Matrices – Test 2021–2022",
        fontsize=15,
    )

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.65,
        label="Anzahl Samples",
    )

    fig.savefig(
        FIGURE_DIR / "confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==================================================
# 7. HAUPTPROGRAMM
# ==================================================

def main():

    print("=" * 60)
    print("WILDFIREFORECAST - FINALE GRAFIKEN")
    print("=" * 60)

    comparison, predictions = load_data()

    print("\nErstelle ROC-Kurven ...")
    plot_roc(predictions)

    print("Erstelle Precision-Recall-Kurven ...")
    plot_precision_recall(predictions)

    print("Erstelle Modellvergleich ...")
    plot_comparison(comparison)

    print("Erstelle Confusion Matrices ...")
    plot_confusion_matrices(predictions)

    print("\n" + "=" * 60)
    print("ALLE VIER GRAFIKEN ERFOLGREICH ERSTELLT")
    print("=" * 60)

    for path in sorted(FIGURE_DIR.glob("*.png")):
        print(path.name)

    print("\nSpeicherort:")
    print(FIGURE_DIR)

    print("\nKeine Modelle trainiert oder getestet.")
    print("Nur gespeicherte Ergebnisse visualisiert.")


if __name__ == "__main__":
    main()
