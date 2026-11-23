import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
SEQ_LEN = 59


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Projekt-Root (4 Ebenen hoch aus src/models/lstm/tuning/)
ROOT = Path(__file__).resolve().parents[4]

# Datenpfad robust erkennen (falls Struktur angepasst wurde)
DATA = ROOT / "data" / "processed" / "provisional"
if not DATA.exists():
    DATA = ROOT / "data" / "processed"

# Zielordner für getunte Modelle (analog zu models/transformer/tuned/)
MODELS_TUNED = ROOT / "models" / "lstm" / "tuned"
MODELS_TUNED.mkdir(parents=True, exist_ok=True)

# Zielordner für Tuning-Berichte (passt sich bestehender Struktur an)
REPORTS = ROOT / "reports" / "lstm" / "tuning"
if not (ROOT / "reports" / "lstm").exists() and (ROOT / "reports" / "tuning_lstm").exists():
    REPORTS = ROOT / "reports" / "tuning_lstm"
REPORTS.mkdir(parents=True, exist_ok=True)

config = json.loads((DATA / "input_config.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(ROOT / "data" / "processed" / "sample_split.csv")


def load_sequence_data(split: str, features: list[str]):
    df = pd.read_csv(DATA / f"{split}_sequences.csv")
    expected = manifest.loc[manifest["split"].eq(split)].set_index("sample_id").sort_index()
    df = df.sort_values(["sample_id", "time_idx"])

    X_2d = df[features].to_numpy(dtype=np.float32)
    y = expected["label"].to_numpy(dtype=np.int64)
    return X_2d, y


class WildfireLSTM(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
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


variants = [
    ("3A", "without_lst", config["features_without_lst"]),
    ("3B", "with_lst", config["features_with_lst"]),
]

param_grid = [
    {"hidden_size": 32, "lr": 0.001, "batch_size": 64, "epochs": 10},
    {"hidden_size": 64, "lr": 0.001, "batch_size": 64, "epochs": 10},
    {"hidden_size": 64, "lr": 0.0005, "batch_size": 32, "epochs": 15},
    {"hidden_size": 128, "lr": 0.001, "batch_size": 64, "epochs": 10},
    {"hidden_size": 128, "lr": 0.001, "batch_size": 64, "epochs": 15},
]

all_tuning_results = []
best_configs = {}

print("\nStarte Hyperparameter-Tuning & Modell-Export für 3A und 3B ...", flush=True)

for model_name, variant, features in variants:
    print(f"\n==================================================")
    print(f" Tune Variante: Modell {model_name} ({variant})")
    print(f"==================================================", flush=True)

    X_train_2d, y_train = load_sequence_data("train", features)
    X_val_2d, y_val = load_sequence_data("validation", features)

    # Preprocessing strikt auf Trainingsdaten (2006-2019) fitten
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_2d))
    X_val_scaled = scaler.transform(imputer.transform(X_val_2d))

    num_features = len(features)
    X_train_3d = X_train_scaled.reshape(-1, SEQ_LEN, num_features)
    X_val_3d = X_val_scaled.reshape(-1, SEQ_LEN, num_features)

    best_roc_auc = -1.0
    best_state_dict = None
    best_params = None

    for i, params in enumerate(param_grid, start=1):
        set_seed(SEED)
        print(f"  -> Lauf {i}/{len(param_grid)} | Parameter: {params}", flush=True)

        train_dataset = TensorDataset(
            torch.tensor(X_train_3d, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        model = WildfireLSTM(num_features=num_features, hidden_size=params["hidden_size"])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        # Training
        model.train()
        for _ in range(params["epochs"]):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # Validierung (2020)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_val_3d, dtype=torch.float32))).numpy()
        preds = (probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, probs)
        avg_prec = average_precision_score(y_val, probs)
        acc = accuracy_score(y_val, preds)
        bal_acc = balanced_accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        all_tuning_results.append({
            "model": model_name,
            "variant": variant,
            "run": i,
            **params,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": avg_prec,
        })

        # Bestes Modell anhand ROC-AUC merken
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_params = params

    # Bestes getuntes Modell (.pt) und Preprocessing (.joblib) speichern
    model_path = MODELS_TUNED / f"lstm_{model_name}_tuned.pt"
    prep_path = MODELS_TUNED / f"lstm_{model_name}_tuned_preprocessing.joblib"

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "num_features": num_features,
            "hidden_size": best_params["hidden_size"],
            "num_layers": 1,
            "dropout": 0.2,
            "seq_len": SEQ_LEN,
            "features": features,
            "best_params": best_params,
            "validation_roc_auc": float(best_roc_auc),
        },
        model_path,
    )

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "features": features,
            "seq_len": SEQ_LEN,
        },
        prep_path,
    )

    best_configs[model_name] = {
        "variant": variant,
        "best_params": best_params,
        "validation_roc_auc": float(best_roc_auc),
        "model_file": str(model_path.relative_to(ROOT)),
        "preprocessing_file": str(prep_path.relative_to(ROOT)),
    }

    print(f"  => Bestes Modell {model_name} gespeichert (ROC-AUC: {best_roc_auc:.4f}): {model_path.name}")

# Ergebnisse speichern
summary_df = pd.DataFrame(all_tuning_results).sort_values(by=["model", "roc_auc"], ascending=[True, False])
summary_df.to_csv(REPORTS / "tuning_summary.csv", index=False)
(REPORTS / "best_params.json").write_text(json.dumps(best_configs, indent=2), encoding="utf-8")

print("\n========================================")
print(" TUNING & EXPORT ABGESCHLOSSEN")
print("========================================")
print(summary_df[["model", "variant", "hidden_size", "lr", "epochs", "roc_auc", "recall", "f1"]].to_string(index=False))
print(f"\nModelle gespeichert in: {MODELS_TUNED.relative_to(ROOT)}")
print(f"Berichte gespeichert in: {REPORTS.relative_to(ROOT)}")
print("Testdaten (2021-2022) wurden unangetastet gelassen.")