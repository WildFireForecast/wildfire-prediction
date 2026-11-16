from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import StandardScaler


# 1. Fester Zufalls-Seed für reproduzierbare Ergebnisse (wichtig für die Dokumentation)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Wichtig: Da wir nun in src/models/lstm/ liegen, müssen wir 3 Ebenen hoch, um zum Projekt-Root zu kommen.
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "provisional"
MODELS = ROOT / "models" / "lstm" / "baseline"
REPORTS = ROOT / "reports" / "lstm/baseline"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

config = json.loads((DATA / "input_config.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(ROOT / "data" / "processed" / "sample_split.csv")

SEQ_LEN = 59  # Exakt 59 Tage Historie pro Sample (time_idx 0 bis 58)


def load_sequence_data(split, features):
    """Lädt die 59-Tage-Sequenzen und prüft die Integrität."""
    df = pd.read_csv(DATA / f"{split}_sequences.csv")

    expected = (
        manifest.loc[manifest["split"].eq(split)]
        .set_index("sample_id")
        .sort_index()
    )

    if set(df["sample_id"]) != set(expected.index):
        raise ValueError(f"{split}: Sample-Zuordnung stimmt nicht mit sample_split.csv überein.")

    df = df.sort_values(["sample_id", "time_idx"])
    sample_ids = df["sample_id"].unique()
    
    if not np.array_equal(sample_ids, expected.index.to_numpy()):
        raise ValueError(f"{split}: Reihenfolge der Sample-IDs stimmt nicht.")

    if len(df) != len(sample_ids) * SEQ_LEN:
        raise ValueError(f"{split}: Nicht jedes Sample hat exakt {SEQ_LEN} Tage.")

    X_2d = df[features].to_numpy(dtype=np.float32)
    if np.isinf(X_2d).any():
        raise ValueError(f"{split}: Unendliche Messwerte vorhanden.")

    y = expected["label"].to_numpy(dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{split}: Ungültige oder fehlende Klassen.")

    return X_2d, y, expected.index


class WildfireLSTM(nn.Module):
    """LSTM-Architektur für die Modelle 3A und 3B."""
    def __init__(self, num_features, hidden_size=64, num_layers=1, dropout=0.2):
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
        # x Form: (Batch, 59 Tage, Anzahl Features)
        lstm_out, _ = self.lstm(x)
        # Wir nehmen den Zustand nach dem allerletzten Tag (Index -1)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        logits = self.fc(out)
        return logits.squeeze(-1)


# Varianten-Definition: 3A (ohne LST) und 3B (mit LST)
variants = [
    ("3A", "without_lst", config["features_without_lst"]),
    ("3B", "with_lst", config["features_with_lst"]),
]

results = []

for model_name, variant, features in variants:
    print(f"\n========================================", flush=True)
    print(f" Trainiere Modell {model_name} ({variant}) ...", flush=True)
    print(f"========================================", flush=True)

    X_train_2d, y_train, train_ids = load_sequence_data("train", features)
    X_val_2d, y_val, val_ids = load_sequence_data("validation", features)

    if not train_ids.intersection(val_ids).empty:
        raise ValueError(f"Modell {model_name}: Samples sind in Training und Validierung enthalten.")

    # Imputation (Median) und Skalierung NUR auf Trainingsdaten anpassen (verhindert Data Leakage)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_2d))
    X_val_scaled = scaler.transform(imputer.transform(X_val_2d))

    # Umformen in 3D-Tensoren für das LSTM: (Samples, 59 Tage, Features)
    num_features = len(features)
    X_train_3d = X_train_scaled.reshape(-1, SEQ_LEN, num_features)
    X_val_3d = X_val_scaled.reshape(-1, SEQ_LEN, num_features)

    train_dataset = TensorDataset(
        torch.tensor(X_train_3d, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    HIDDEN_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.001

    model = WildfireLSTM(num_features=num_features, hidden_size=HIDDEN_SIZE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)

        avg_loss = epoch_loss / len(train_dataset)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoche {epoch:2d}/{EPOCHS} | Trainings-Loss: {avg_loss:.4f}", flush=True)

    # Evaluation auf dem Validierungsjahr 2020
    model.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val_3d, dtype=torch.float32)
        val_logits = model(val_tensor)
        probabilities = torch.sigmoid(val_logits).numpy()

    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "model": model_name,
        "variant": variant,
        "features": num_features,
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "average_precision": average_precision_score(y_val, probabilities),
    })

    # Abspeichern von Modell, Preprocessing und Ergebnissen
    torch.save({
        "model_state_dict": model.state_dict(),
        "features": features,
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
    }, MODELS / f"lstm_{model_name}.pt")
    
    joblib.dump({"imputer": imputer, "scaler": scaler}, MODELS / f"lstm_{model_name}_preprocessing.joblib")

    pd.DataFrame({
        "label": y_val,
        "score": probabilities,
        "prediction": predictions,
    }, index=val_ids).to_csv(REPORTS / f"{model_name}_validation_predictions.csv")

    matrix = pd.DataFrame(
        confusion_matrix(y_val, predictions, labels=[0, 1]),
        index=["tatsaechlich_0", "tatsaechlich_1"],
        columns=["vorhergesagt_0", "vorhergesagt_1"],
    )
    matrix.to_csv(REPORTS / f"{model_name}_confusion_matrix.csv")

    print(f"\nVerwechslungsmatrix für Modell {model_name}:")
    print(matrix.to_string())


# Gesamttabelle der Validierungsergebnisse speichern
summary = pd.DataFrame(results).set_index("model")
summary.to_csv(REPORTS / "validation_metrics.csv")

run_info = {
    "status": "provisional",
    "target_day_assumption": config["target_day_assumption"],
    "verification": config["verification"],
    "torch_version": torch.__version__,
    "sklearn_version": sklearn.__version__,
    "training_years": "2006–2019",
    "validation_year": 2020,
    "test_used": False,
    "sequence_length_days": SEQ_LEN,
    "hidden_size": 64,
    "epochs": 15,
    "learning_rate": 0.001,
    "threshold": 0.5,
    "imputation": "Trainingsmedian pro Merkmal über alle Trainingstage",
    "scaling": "StandardScaler, ausschliesslich auf Training angepasst",
}

(REPORTS / "run_info.json").write_text(
    json.dumps(run_info, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n========================================")
print(" VORLÄUFIGE ERGEBNISSE: VALIDIERUNG 2020")
print("========================================")
print(summary.round(4).to_string())
print("\nModelle gespeichert in: models/lstm/baseline/")
print("Berichte gespeichert in: reports/lstm/baseline/")
print("Testdaten (2021-2022) wurden unangetastet gelassen.")