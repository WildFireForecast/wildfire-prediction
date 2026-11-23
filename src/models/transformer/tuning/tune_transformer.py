
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ==================================================
# 1. EINSTELLUNGEN
# ==================================================

SEED = 42
SEQ_LEN = 59
EPOCHS = 15
THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "provisional"
MANIFEST_PATH = ROOT / "data" / "processed" / "sample_split.csv"

REPORTS = ROOT / "reports" / "transformer" / "tuning"
MODELS = ROOT / "models" / "transformer" / "tuned"

REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Verwendetes Geraet: {device}")

config = json.loads(
    (DATA / "input_config.json").read_text(encoding="utf-8")
)

manifest = pd.read_csv(MANIFEST_PATH)

if manifest["sample_id"].duplicated().any():
    raise ValueError("Doppelte Sample-IDs im Manifest.")

# ==================================================
# 2. VARIANTEN UND TUNING-KONFIGURATIONEN
# ==================================================

variants = [
    ("4A", config["features_without_lst"]),
    ("4B", config["features_with_lst"]),
]

param_grid = [
    {
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "lr": 0.001,
        "batch_size": 64,
    },
    {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "lr": 0.001,
        "batch_size": 64,
    },
    {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "lr": 0.0005,
        "batch_size": 32,
    },
    {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "lr": 0.0005,
        "batch_size": 64,
    },
]

# ==================================================
# 3. DATEN LADEN
# ==================================================

def load_sequence_data(split, features):

    if split not in {"train", "validation"}:
        raise ValueError("Testdaten sind hier nicht erlaubt.")

    df = pd.read_csv(DATA / f"{split}_sequences.csv")

    expected = (
        manifest.loc[manifest["split"].eq(split)]
        .set_index("sample_id")
        .sort_index()
    )

    if set(df["sample_id"]) != set(expected.index):
        raise ValueError(f"{split}: Sample-IDs stimmen nicht.")

    df = df.sort_values(["sample_id", "time_idx"])
    ids = df["sample_id"].unique()

    if not np.array_equal(ids, expected.index.to_numpy()):
        raise ValueError(f"{split}: Reihenfolge stimmt nicht.")

    if len(df) != len(ids) * SEQ_LEN:
        raise ValueError(f"{split}: Ungueltige Sequenzlaenge.")

    valid_steps = df.groupby("sample_id")["time_idx"].apply(
        lambda x: np.array_equal(x.to_numpy(), np.arange(SEQ_LEN))
    )

    if not valid_steps.all():
        raise ValueError(f"{split}: Zeitindizes fehlerhaft.")

    X = df[features].to_numpy(dtype=np.float32)
    y = expected["label"].to_numpy(dtype=np.int64)

    if np.isinf(X).any():
        raise ValueError(f"{split}: Unendliche Werte.")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{split}: Ungueltige Labels.")

    return X, y, expected.index


# ==================================================
# 4. TRANSFORMER-ARCHITEKTUR
# Identisch mit der Baseline
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

        nn.init.normal_(self.pos_embedding, std=0.02)

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
# 5. TUNING DURCHFUEHREN
# ==================================================

all_results = []

for model_name, features in variants:

    print("\n========================================")
    print(f" TUNING TRANSFORMER {model_name}")
    print("========================================", flush=True)

    X_train, y_train, train_ids = load_sequence_data(
        "train", features
    )
    X_val, y_val, val_ids = load_sequence_data(
        "validation", features
    )

    if not train_ids.intersection(val_ids).empty:
        raise ValueError("Data Leakage: Gleiche Sample-IDs!")

    num_features = len(features)

    # Preprocessing nur auf Trainingsdaten fitten
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        imputer.fit_transform(X_train)
    )
    X_val = scaler.transform(
        imputer.transform(X_val)
    )

    if X_train.shape[1] != num_features:
        raise ValueError("Falsche Anzahl Trainingsfeatures.")

    if not np.isfinite(X_train).all():
        raise ValueError("Ungueltige Trainingswerte.")

    if not np.isfinite(X_val).all():
        raise ValueError("Ungueltige Validierungswerte.")

    X_train = X_train.reshape(
        -1, SEQ_LEN, num_features
    )
    X_val = X_val.reshape(
        -1, SEQ_LEN, num_features
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_tensor = torch.tensor(
        X_val, dtype=torch.float32
    ).to(device)

    best_auc = -1.0
    best_run = None

    for run, params in enumerate(param_grid, start=1):

        print(f"\n{model_name}: Lauf {run}/{len(param_grid)}")
        print(f"Parameter: {params}", flush=True)

        # Gleicher Seed fuer jeden Trainingslauf
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            generator=generator,
        )

        model = WildfireTransformer(
            num_features=num_features,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
        )

        criterion = nn.BCEWithLogitsLoss()

        history = []

        for epoch in range(1, EPOCHS + 1):

            model.train()
            total_loss = 0.0

            for batch_x, batch_y in train_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)

            avg_loss = total_loss / len(train_dataset)

            history.append({
                "model": model_name,
                "run": run,
                "epoch": epoch,
                "train_loss": avg_loss,
            })

            if epoch in {1, 5, 10, 15}:
                print(
                    f"Epoche {epoch:02d}/{EPOCHS} "
                    f"| Loss: {avg_loss:.4f}",
                    flush=True,
                )

        # Validierung auf 2020
        model.eval()

        with torch.no_grad():
            logits = model(val_tensor)
            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        auc = roc_auc_score(y_val, probabilities)

        result = {
            "model": model_name,
            "run": run,
            **params,
            "epochs": EPOCHS,
            "accuracy": accuracy_score(y_val, predictions),
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "roc_auc": auc,
            "average_precision": average_precision_score(
                y_val, probabilities
            ),
        }

        all_results.append(result)

        print(
            f"ROC-AUC: {auc:.4f} | "
            f"F1: {result['f1']:.4f}",
            flush=True,
        )

        # Alle Trainingsverlaeufe speichern
        history_path = (
            REPORTS / f"{model_name}_run_{run}_history.csv"
        )
        pd.DataFrame(history).to_csv(
            history_path, index=False
        )

        # Bestes Modell anhand Validierungs-ROC-AUC
        if auc > best_auc:

            best_auc = auc
            best_run = run

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": features,
                    "seq_len": SEQ_LEN,
                    "d_model": params["d_model"],
                    "nhead": params["nhead"],
                    "num_layers": params["num_layers"],
                    "dropout": 0.2,
                },
                MODELS / f"transformer_{model_name}_tuned.pt",
            )

            joblib.dump(
                {
                    "imputer": imputer,
                    "scaler": scaler,
                    "features": features,
                },
                MODELS / f"transformer_{model_name}_tuned_preprocessing.joblib",
            )

            pd.DataFrame(
                {
                    "label": y_val,
                    "score": probabilities,
                    "prediction": predictions,
                },
                index=val_ids,
            ).to_csv(
                REPORTS / f"{model_name}_best_validation_predictions.csv"
            )

            pd.DataFrame(
                confusion_matrix(
                    y_val, predictions, labels=[0, 1]
                ),
                index=["tatsaechlich_0", "tatsaechlich_1"],
                columns=["vorhergesagt_0", "vorhergesagt_1"],
            ).to_csv(
                REPORTS / f"{model_name}_best_confusion_matrix.csv"
            )

            (REPORTS / f"{model_name}_best_params.json").write_text(
                json.dumps(
                    {
                        "model": model_name,
                        "best_run": run,
                        "best_validation_roc_auc": auc,
                        "parameters": params,
                        "epochs": EPOCHS,
                        "target_day_assumption": config[
                            "target_day_assumption"
                        ],
                        "status": "provisional",
                        "test_used": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    print(
        f"\nBestes {model_name}: "
        f"Lauf {best_run}, ROC-AUC {best_auc:.4f}",
        flush=True,
    )

# ==================================================
# 6. GESAMTERGEBNIS SPEICHERN
# ==================================================

summary = pd.DataFrame(all_results)

summary = summary.sort_values(
    by=["model", "roc_auc"],
    ascending=[True, False],
)

summary.to_csv(
    REPORTS / "tuning_summary.csv",
    index=False,
)

print("\n========================================")
print(" TRANSFORMER-TUNING ABGESCHLOSSEN")
print("========================================")
print(summary.to_string(index=False))
print("\nErgebnisse:", REPORTS)
print("Modelle:", MODELS)
print("Testdaten wurden nicht verwendet.")
