
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from metrics import evaluate_predictions
from test_guard import require_final_test_approval


# ==================================================
# 1. PROJEKTPFADE
# ==================================================

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "processed" / "provisional"
MANIFEST_PATH = ROOT / "data" / "processed" / "sample_split.csv"
REPORT_DIR = ROOT / "reports" / "final_test"

SEQ_LEN = 59
DEVICE = torch.device("cpu")


# ==================================================
# 2. AUSGEWÄHLTE MODELLVERSIONEN
# ==================================================

MODELS = {
    "3A": {
        "model": ROOT / "models/lstm/baseline/lstm_3A.pt",
        "preprocessing": ROOT
        / "models/lstm/baseline/lstm_3A_preprocessing.joblib",
        "features": 25,
        "version": "baseline",
    },
    "3B": {
        "model": ROOT / "models/lstm/tuned/lstm_3B_tuned.pt",
        "preprocessing": ROOT
        / "models/lstm/tuned/lstm_3B_tuned_preprocessing.joblib",
        "features": 27,
        "version": "tuned",
    },
}


# ==================================================
# 3. LSTM-ARCHITEKTUR
# ==================================================

class WildfireLSTM(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


# ==================================================
# 4. MODELL UND VORVERARBEITUNG LADEN
# ==================================================

def load_model(name, settings):

    checkpoint = torch.load(
        settings["model"],
        map_location=DEVICE,
        weights_only=True,
    )

    preprocessing = joblib.load(
        settings["preprocessing"]
    )

    features = list(checkpoint["features"])

    num_features = len(features)
    seq_len = int(checkpoint["seq_len"])

    if num_features != settings["features"]:
        raise ValueError(
            f"{name}: Falsche Feature-Anzahl."
        )

    if seq_len != SEQ_LEN:
        raise ValueError(
            f"{name}: Falsche Sequenzlänge."
        )

    if "features" in preprocessing:
        if list(preprocessing["features"]) != features:
            raise ValueError(
                f"{name}: Feature-Reihenfolge stimmt nicht."
            )

    if "seq_len" in preprocessing:
        if int(preprocessing["seq_len"]) != seq_len:
            raise ValueError(
                f"{name}: Sequenzlängen stimmen nicht."
            )

    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]

    if not hasattr(imputer, "transform"):
        raise TypeError(f"{name}: Imputer ungültig.")

    if not hasattr(scaler, "transform"):
        raise TypeError(f"{name}: Scaler ungültig.")

    if scaler.n_features_in_ != num_features:
        raise ValueError(
            f"{name}: Scaler-Feature-Anzahl stimmt nicht."
        )

    model = WildfireLSTM(
        num_features=num_features,
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint.get("num_layers", 1)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )

    # Strikte Kontrolle der gespeicherten Modellgewichte
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model, preprocessing, features, checkpoint


# ==================================================
# 5. MODELLE OHNE TESTDATEN PRÜFEN
# ==================================================

def check_models():

    print("=" * 60)
    print("LSTM - MODELLPRÜFUNG")
    print("=" * 60)

    for name, settings in MODELS.items():

        print(f"\nPrüfe LSTM {name} ...", flush=True)

        model, prep, features, checkpoint = load_model(
            name, settings
        )

        # Künstliche Eingabe: keine echten Daten
        dummy = torch.zeros(
            2, SEQ_LEN, len(features),
            dtype=torch.float32,
        )

        with torch.no_grad():
            output = model(dummy)

        if output.shape != (2,):
            raise ValueError(
                f"{name}: Unerwartete Ausgabeform."
            )

        if not torch.isfinite(output).all():
            raise ValueError(
                f"{name}: Ungültige Modellausgabe."
            )

        print("Checkpoint geladen: OK")
        print("Modellgewichte rekonstruiert: OK")
        print("Preprocessing: OK")
        print("Vorhersagefunktion: OK")
        print("Features:", len(features))
        print("Sequenzlänge:", SEQ_LEN)
        print("Hidden Size:", checkpoint["hidden_size"])

    print("\nBeide LSTM-Modelle erfolgreich geprüft.")
    print("Keine Testdaten verwendet.")


# ==================================================
# 6. TESTDATEN LADEN
# NUR NACH WISSENSCHAFTLICHER FREIGABE
# ==================================================

def load_test_data(features):

    require_final_test_approval()

    manifest = pd.read_csv(MANIFEST_PATH)

    expected = (
        manifest.loc[manifest["split"].eq("test")]
        .set_index("sample_id")
        .sort_index()
    )

    if expected.empty or expected.index.has_duplicates:
        raise ValueError(
            "Ungültiges Test-Sample-Manifest."
        )

    if not expected["label"].isin([0, 1]).all():
        raise ValueError("Ungültige Testlabels.")

    if expected["label"].nunique() != 2:
        raise ValueError(
            "Im Test müssen beide Klassen vorhanden sein."
        )

    df = pd.read_csv(
        DATA_DIR / "test_sequences.csv"
    )

    if df.duplicated(["sample_id", "time_idx"]).any():
        raise ValueError(
            "Doppelte Zeitindizes in Testsequenzen."
        )

    if set(df["sample_id"]) != set(expected.index):
        raise ValueError(
            "Test-Sample-IDs stimmen nicht überein."
        )

    df = df.sort_values(
        ["sample_id", "time_idx"]
    )

    sample_ids = df["sample_id"].unique()

    if not np.array_equal(
        sample_ids, expected.index.to_numpy()
    ):
        raise ValueError(
            "Reihenfolge der Test-Samples stimmt nicht."
        )

    valid_steps = df.groupby("sample_id")["time_idx"].apply(
        lambda values: np.array_equal(
            values.to_numpy(),
            np.arange(SEQ_LEN),
        )
    )

    if not valid_steps.all():
        raise ValueError(
            "Mindestens eine Testsequenz ist unvollständig."
        )

    if not set(features).issubset(df.columns):
        raise ValueError(
            "Mindestens ein Eingabefeature fehlt."
        )

    X = df[features].to_numpy(dtype=np.float32)

    if np.isinf(X).any():
        raise ValueError(
            "Unendliche Feature-Werte gefunden."
        )

    y = expected["label"].to_numpy(dtype=int)

    return X, y, expected.index


# ==================================================
# 7. FINALE TESTEVALUATION
# ==================================================

def run_final_evaluation():

    require_final_test_approval()

    if "--execute" not in sys.argv:
        raise RuntimeError(
            "Für die Testausführung ist --execute erforderlich."
        )

    print("=" * 60)
    print("LSTM - FINALE TESTEVALUATION")
    print("=" * 60)

    results = []
    prediction_files = {}

    reference_ids = None
    reference_labels = None

    for name, settings in MODELS.items():

        print(f"\nEvaluiere LSTM {name} ...", flush=True)

        model, prep, features, checkpoint = load_model(
            name, settings
        )

        X, y, sample_ids = load_test_data(features)

        if reference_ids is None:
            reference_ids = sample_ids
            reference_labels = y.copy()
        else:
            if not sample_ids.equals(reference_ids):
                raise ValueError(
                    "Test-Sample-IDs unterscheiden sich."
                )

            if not np.array_equal(y, reference_labels):
                raise ValueError(
                    "Testlabels unterscheiden sich."
                )

        # Gespeicherte Vorverarbeitung verwenden.
        # Kein fit() und kein erneutes Training.
        X_imputed = prep["imputer"].transform(X)
        X_scaled = prep["scaler"].transform(X_imputed)

        if X_scaled.shape[1] != len(features):
            raise ValueError(
                f"{name}: Feature-Anzahl nach Skalierung falsch."
            )

        if not np.isfinite(X_scaled).all():
            raise ValueError(
                f"{name}: Ungültige skalierte Werte."
            )

        X_3d = X_scaled.reshape(
            len(y), SEQ_LEN, len(features)
        )

        tensor = torch.tensor(
            X_3d,
            dtype=torch.float32,
            device=DEVICE,
        )

        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            scores = torch.sigmoid(logits).cpu().numpy()

        metrics = evaluate_predictions(
            name, y, scores
        )

        metrics["version"] = settings["version"]
        results.append(metrics)

        predictions = (scores >= 0.5).astype(int)

        predictions_df = pd.DataFrame(
            {
                "label": y,
                "score": scores,
                "prediction": predictions,
            },
            index=sample_ids,
        )

        predictions_df.index.name = "sample_id"
        prediction_files[name] = predictions_df

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

    # Ergebnisse erst speichern, wenn beide Modelle
    # erfolgreich ausgewertet wurden.
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in prediction_files.items():
        df.to_csv(
            REPORT_DIR / f"{name}_test_predictions.csv"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "lstm_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("LSTM - FINALE TESTEVALUATION ABGESCHLOSSEN")
    print("=" * 60)

    print(results_df.to_string(index=False))
    print("\nGespeichert unter:", REPORT_DIR)


# ==================================================
# 8. PROGRAMMSTART
# ==================================================

def main():

    if "--execute" not in sys.argv:
        check_models()
        return

    run_final_evaluation()


if __name__ == "__main__":
    main()
