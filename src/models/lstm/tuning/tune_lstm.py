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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# 1. Reproduzierbarkeit sicherstellen
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Pfade anpassen (4 Ebenen hoch zum Projekt-Root)
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "processed" / "provisional"
REPORTS = ROOT / "reports" / "tuning_lstm"
REPORTS.mkdir(parents=True, exist_ok=True)

config = json.loads((DATA / "input_config.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(ROOT / "data" / "processed" / "sample_split.csv")

SEQ_LEN = 59

def load_sequence_data(split, features):
    df = pd.read_csv(DATA / f"{split}_sequences.csv")
    expected = manifest.loc[manifest["split"].eq(split)].set_index("sample_id").sort_index()
    df = df.sort_values(["sample_id", "time_idx"])
    
    X_2d = df[features].to_numpy(dtype=np.float32)
    y = expected["label"].to_numpy(dtype=np.int64)
    return X_2d, y


class WildfireLSTM(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        return self.fc(out).squeeze(-1)


# Wir definieren die beiden Modellvarianten 3A und 3B
variants = [
    ("3A", "without_lst", config["features_without_lst"]),
    ("3B", "with_lst", config["features_with_lst"]),
]

# Das Raster an Hyperparametern, die wir testen wollen
param_grid = [
    {"hidden_size": 32, "lr": 0.001, "batch_size": 64, "epochs": 10},
    {"hidden_size": 64, "lr": 0.001, "batch_size": 64, "epochs": 10},
    {"hidden_size": 64, "lr": 0.0005, "batch_size": 32, "epochs": 15},
    {"hidden_size": 128, "lr": 0.001, "batch_size": 64, "epochs": 10},
]

all_tuning_results = []

print("\nStarte systematisches Hyperparameter-Tuning für Modell 3A und 3B...", flush=True)

for model_name, variant, features in variants:
    print(f"\n==================================================")
    print(f" Tunue Variante: Modell {model_name} ({variant})")
    print(f"==================================================", flush=True)

    X_train_2d, y_train = load_sequence_data("train", features)
    X_val_2d, y_val = load_sequence_data("validation", features)

    # Preprocessing (Median-Imputation & Skalierung nur auf Training anpassen)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_2d))
    X_val_scaled = scaler.transform(imputer.transform(X_val_2d))

    num_features = len(features)
    X_train_3d = X_train_scaled.reshape(-1, SEQ_LEN, num_features)
    X_val_3d = X_val_scaled.reshape(-1, SEQ_LEN, num_features)

    for i, params in enumerate(param_grid, start=1):
        print(f"  -> Lauf {i}/{len(param_grid)} | Parameter: {params}", flush=True)
        
        train_dataset = TensorDataset(torch.tensor(X_train_3d, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        model = WildfireLSTM(num_features=num_features, hidden_size=params["hidden_size"])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

        # Training
        model.train()
        for epoch in range(params["epochs"]):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # Validierung auf 2020
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_val_3d, dtype=torch.float32))).numpy()
        preds = (probs >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, probs)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, zero_division=0)

        all_tuning_results.append({
            "model": model_name,
            "variant": variant,
            "run": i,
            **params,
            "roc_auc": roc_auc,
            "accuracy": acc,
            "f1": f1,
        })

# Gesamtergebnis speichern
summary_df = pd.DataFrame(all_tuning_results).sort_values(by=["model", "roc_auc"], ascending=[True, False])
summary_df.to_csv(REPORTS / "tuning_summary.csv", index=False)

print("\n========================================")
print(" TUNING FÜR 3A & 3B ABGESCHLOSSEN")
print("========================================")
print(summary_df.to_string(index=False))
print(f"\nGespeichert unter: reports/tuning_lstm/tuning_summary.csv")