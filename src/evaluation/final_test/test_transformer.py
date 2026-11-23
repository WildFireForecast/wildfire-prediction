
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

DATA_DIR = ROOT / "data/processed/provisional"
MANIFEST_PATH = ROOT / "data/processed/sample_split.csv"
MODEL_DIR = ROOT / "models/transformer/tuned"
REPORT_DIR = ROOT / "reports/final_test"

SEQ_LEN = 59
DEVICE = torch.device("cpu")


# ==================================================
# 2. AUSGEWÄHLTE TRANSFORMER-MODELLE
# ==================================================

MODELS = {
    "4A": {
        "model": MODEL_DIR / "transformer_4A_tuned.pt",
        "preprocessing": (
            MODEL_DIR
            / "transformer_4A_tuned_preprocessing.joblib"
        ),
        "expected_features": 25,
        "version": "tuned",
    },
    "4B": {
        "model": MODEL_DIR / "transformer_4B_tuned.pt",
        "preprocessing": (
            MODEL_DIR
            / "transformer_4B_tuned_preprocessing.joblib"
        ),
        "expected_features": 27,
        "version": "tuned",
    },
}


# ==================================================
# 3. TRANSFORMER-ARCHITEKTUR
# IDENTISCH MIT DEM TRAININGSSKRIPT
# ==================================================

class WildfireTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            num_features, d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, SEQ_LEN, d_model)
        )

        nn.init.normal_(
            self.pos_embedding, std=0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):

        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.classifier(x).squeeze(-1)


# ==================================================
# 4. MODELL UND PREPROCESSING LADEN
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

    if len(features) != settings["expected_features"]:
        raise ValueError(
            f"{name}: Feature-Anzahl stimmt nicht."
        )

    if int(checkpoint["seq_len"]) != SEQ_LEN:
        raise ValueError(
            f"{name}: Ungültige Sequenzlänge."
        )

    if list(preprocessing["features"]) != features:
        raise ValueError(
            f"{name}: Feature-Reihenfolge stimmt nicht."
        )

    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]

    if not hasattr(imputer, "transform"):
        raise TypeError(
            f"{name}: Imputer ungültig."
        )

    if not hasattr(scaler, "transform"):
        raise TypeError(
            f"{name}: Scaler ungültig."
        )

    if scaler.n_features_in_ != len(features):
        raise ValueError(
            f"{name}: Scaler-Feature-Anzahl falsch."
        )

    d_model = int(checkpoint["d_model"])
    nhead = int(checkpoint["nhead"])
    num_layers = int(checkpoint["num_layers"])

    if d_model <= 0 or nhead <= 0 or num_layers <= 0:
        raise ValueError(
            f"{name}: Ungültige Architektur."
        )

    if d_model % nhead != 0:
        raise ValueError(
            f"{name}: Embedding nicht durch Heads teilbar."
        )

    model = WildfireTransformer(
        num_features=len(features),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=float(checkpoint.get("dropout", 0.2)),
    )

    # Gespeicherte Gewichte vollständig rekonstruieren
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model, preprocessing, features, checkpoint


# ==================================================
# 5. SICHERE MODELLPRÜFUNG OHNE TESTDATEN
# ==================================================

def check_models():

    print("=" * 60)
    print("TRANSFORMER - MODELLPRÜFUNG")
    print("=" * 60)

    for name, settings in MODELS.items():

        print(f"\nPrüfe Transformer {name} ...", flush=True)

        model, prep, features, checkpoint = load_model(
            name, settings
        )

        # Künstliche Eingabedaten
        dummy = torch.zeros(
            2,
            SEQ_LEN,
            len(features),
            dtype=torch.float32,
        )

        with torch.no_grad():
            output = model(dummy)

        if output.shape != (2,):
            raise ValueError(
                f"{name}: Falsche Ausgabeform."
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
        print("Embedding:", checkpoint["d_model"])
        print("Attention Heads:", checkpoint["nhead"])
        print("Encoder Layers:", checkpoint["num_layers"])

    print("\nBeide Transformer erfolgreich geprüft.")
    print("Keine Testdaten verwendet.")


# ==================================================
# 6. TESTSEQUENZEN LADEN
# NUR NACH WISSENSCHAFTLICHER FREIGABE
# ==================================================

def load_test_data(features):

    require_final_test_approval()

    manifest = pd.read_csv(MANIFEST_PATH)

    if manifest["sample_id"].duplicated().any():
        raise ValueError(
            "Doppelte Sample-IDs im Manifest."
        )

    expected = (
        manifest.loc[manifest["split"].eq("test")]
        .set_index("sample_id")
        .sort_index()
    )

    if expected.empty:
        raise ValueError(
            "Keine Testsamples gefunden."
        )

    if not expected["label"].isin([0, 1]).all():
        raise ValueError(
            "Ungültige Testlabels."
        )

    if expected["label"].nunique() != 2:
        raise ValueError(
            "Im Test müssen beide Klassen vorhanden sein."
        )

    df = pd.read_csv(
        DATA_DIR / "test_sequences.csv"
    )

    if df.duplicated(
        ["sample_id", "time_idx"]
    ).any():
        raise ValueError(
            "Doppelte Zeitindizes innerhalb eines Samples."
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
            "Sample-Reihenfolge stimmt nicht."
        )

    valid_steps = df.groupby("sample_id")["time_idx"].apply(
        lambda values: np.array_equal(
            values.to_numpy(),
            np.arange(SEQ_LEN),
        )
    )

    if not valid_steps.all():
        raise ValueError(
            "Mindestens eine Sequenz hat nicht 59 Tage."
        )

    if not set(features).issubset(df.columns):
        raise ValueError(
            "Mindestens ein Feature fehlt."
        )

    X = df[features].to_numpy(
        dtype=np.float32
    )

    if np.isinf(X).any():
        raise ValueError(
            "Unendliche Feature-Werte gefunden."
        )

    y = expected["label"].to_numpy(
        dtype=int
    )

    return X, y, expected.index


# ==================================================
# 7. FINALE TESTEVALUATION
# ==================================================

def run_final_evaluation():

    require_final_test_approval()

    if "--execute" not in sys.argv:
        raise RuntimeError(
            "Für den Test ist --execute erforderlich."
        )

    print("=" * 60)
    print("TRANSFORMER - FINALE TESTEVALUATION")
    print("=" * 60)

    results = []
    prediction_files = {}

    reference_ids = None
    reference_labels = None

    for name, settings in MODELS.items():

        print(
            f"\nEvaluiere Transformer {name} ...",
            flush=True,
        )

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

        # Gespeicherte Vorverarbeitung
        # ausschliesslich mit transform() anwenden.
        X_imputed = prep["imputer"].transform(X)
        X_scaled = prep["scaler"].transform(
            X_imputed
        )

        if X_scaled.shape[1] != len(features):
            raise ValueError(
                f"{name}: Feature-Anzahl falsch."
            )

        if not np.isfinite(X_scaled).all():
            raise ValueError(
                f"{name}: Ungültige Werte nach Skalierung."
            )

        X_3d = X_scaled.reshape(
            len(y),
            SEQ_LEN,
            len(features),
        )

        # Batchweise Vorhersage zur Speicherbegrenzung
        scores_list = []

        model.eval()

        with torch.no_grad():

            for start in range(0, len(y), 64):

                batch = torch.tensor(
                    X_3d[start:start + 64],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                logits = model(batch)

                probabilities = torch.sigmoid(
                    logits
                ).cpu().numpy()

                scores_list.append(
                    np.atleast_1d(probabilities)
                )

        scores = np.concatenate(scores_list)

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

        print(
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )
        print(
            f"F1-Score: {metrics['f1']:.4f}"
        )

    # Ergebnisse erst nach erfolgreicher
    # Auswertung beider Varianten speichern.
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
        REPORT_DIR / "transformer_test_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("TRANSFORMER - TEST ABGESCHLOSSEN")
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
