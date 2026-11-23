
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
from sklearn.preprocessing import StandardScaler
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


# ==================================================
# 1. KONFIGURATION UND REPRODUZIERBARKEIT
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data" / "processed" / "provisional"
MODELS = ROOT / "models" / "transformer" / "baseline"
REPORTS = ROOT / "reports" / "transformer" / "baseline"

MODEL_NAME = "4B"
SEQ_LEN = 59

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.2
THRESHOLD = 0.5

config = json.loads(
    (DATA / "input_config.json").read_text(encoding="utf-8")
)

# Modell 4B: MIT Landoberflaechentemperatur
FEATURES = config["features_with_lst"]
NUM_FEATURES = len(FEATURES)

assert NUM_FEATURES == 27
assert "lst_day" in FEATURES
assert "lst_night" in FEATURES

print("\n========================================")
print(" TRANSFORMER MODELL 4B - MIT LST")
print("========================================")
print(f"Modell: {MODEL_NAME}")
print(f"Anzahl Features: {NUM_FEATURES}")
print(f"Sequenzlaenge: {SEQ_LEN} Tage")
print("Training: 2006-2019")
print("Validierung: 2020")
print("Testdaten werden nicht verwendet.")


# ==================================================
# 2. DATEN LADEN UND UEBERPRUEFEN
# ==================================================

manifest = pd.read_csv(
    ROOT / "data" / "processed" / "sample_split.csv"
)

if manifest["sample_id"].duplicated().any():
    raise ValueError("Doppelte Sample-IDs im Manifest.")


def load_sequence_data(split, features):
    """Laedt 59-Tage-Sequenzen mit Integritaetspruefung."""

    if split not in {"train", "validation"}:
        raise ValueError(
            "Nur Training und Validierung sind erlaubt."
        )

    df = pd.read_csv(
        DATA / f"{split}_sequences.csv"
    )

    expected = (
        manifest.loc[manifest["split"].eq(split)]
        .set_index("sample_id")
        .sort_index()
    )

    if set(df["sample_id"]) != set(expected.index):
        raise ValueError(
            f"{split}: Sample-IDs stimmen nicht ueberein."
        )

    df = df.sort_values(["sample_id", "time_idx"])
    sample_ids = df["sample_id"].unique()

    if not np.array_equal(
        sample_ids, expected.index.to_numpy()
    ):
        raise ValueError(
            f"{split}: Reihenfolge der Samples stimmt nicht."
        )

    if len(df) != len(sample_ids) * SEQ_LEN:
        raise ValueError(
            f"{split}: Falsche Anzahl Zeitschritte."
        )

    valid_indices = df.groupby("sample_id")["time_idx"].apply(
        lambda x: np.array_equal(
            x.to_numpy(), np.arange(SEQ_LEN)
        )
    )

    if not valid_indices.all():
        raise ValueError(
            f"{split}: Zeitindizes sind unvollstaendig."
        )

    if not set(features).issubset(df.columns):
        raise ValueError(
            f"{split}: Mindestens ein Feature fehlt."
        )

    X = df[features].to_numpy(dtype=np.float32)
    y = expected["label"].to_numpy(dtype=np.int64)

    if not np.isfinite(y).all():
        raise ValueError("Ungueltige Labels gefunden.")

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels muessen 0 oder 1 sein.")

    if np.isinf(X).any():
        raise ValueError(
            f"{split}: Unendliche Feature-Werte gefunden."
        )

    return X, y, expected.index


X_train, y_train, train_ids = load_sequence_data(
    "train", FEATURES
)

X_val, y_val, val_ids = load_sequence_data(
    "validation", FEATURES
)

if not train_ids.intersection(val_ids).empty:
    raise ValueError(
        "Data Leakage: Gleiche Sample-IDs in beiden Splits!"
    )

print("\nDaten erfolgreich geladen!")
print(f"Trainingssamples: {len(y_train)}")
print(f"Validierungssamples: {len(y_val)}")
print(f"Features: {NUM_FEATURES}")
print("Keine Ueberschneidung der Sample-IDs.")


# ==================================================
# 3. PREPROCESSING OHNE DATA LEAKAGE
# ==================================================

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imputed
)

X_val_scaled = scaler.transform(
    X_val_imputed
)

if X_train_scaled.shape[1] != NUM_FEATURES:
    raise ValueError("Trainingsfeatures stimmen nicht.")

if X_val_scaled.shape[1] != NUM_FEATURES:
    raise ValueError("Validierungsfeatures stimmen nicht.")

if not np.isfinite(X_train_scaled).all():
    raise ValueError("Ungueltige Trainingswerte.")

if not np.isfinite(X_val_scaled).all():
    raise ValueError("Ungueltige Validierungswerte.")

X_train_3d = X_train_scaled.reshape(
    -1, SEQ_LEN, NUM_FEATURES
)

X_val_3d = X_val_scaled.reshape(
    -1, SEQ_LEN, NUM_FEATURES
)

print("\nPreprocessing erfolgreich!")
print(f"Trainingsdaten: {X_train_3d.shape}")
print(f"Validierungsdaten: {X_val_3d.shape}")
print("Imputer und Scaler nur auf Training angepasst.")


# ==================================================
# 4. TRANSFORMER ARCHITEKTUR
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

        logits = self.classifier(x)

        return logits.squeeze(-1)


model = WildfireTransformer(
    num_features=NUM_FEATURES,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

num_parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nTransformer 4B erfolgreich erstellt!")
print(f"Trainierbare Parameter: {num_parameters}")

# Architektur ueberpruefen
model.eval()

dummy_input = torch.zeros(
    2, SEQ_LEN, NUM_FEATURES
)

with torch.no_grad():
    dummy_output = model(dummy_input)

assert dummy_output.shape == (2,)

print("Eingabeform:", tuple(dummy_input.shape))
print("Ausgabeform:", tuple(dummy_output.shape))
print("Architekturpruefung erfolgreich!")


# ==================================================
# 5. TRAINING VON MODELL 4B
# ==================================================

train_dataset = TensorDataset(
    torch.tensor(X_train_3d, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

training_history = []

print("\n========================================")
print(" STARTE TRAINING: TRANSFORMER 4B")
print("========================================")

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    avg_loss = total_loss / len(train_dataset)

    training_history.append({
        "epoch": epoch,
        "train_loss": avg_loss,
    })

    print(
        f"Epoche {epoch:02d}/{EPOCHS} "
        f"| Trainings-Loss: {avg_loss:.4f}",
        flush=True,
    )

print("\nTraining von Modell 4B abgeschlossen!")


# ==================================================
# 6. VALIDIERUNG AUF DEM JAHR 2020
# ==================================================

model.eval()

val_tensor = torch.tensor(
    X_val_3d, dtype=torch.float32
)

with torch.no_grad():

    val_logits = model(val_tensor)

    probabilities = torch.sigmoid(
        val_logits
    ).cpu().numpy()

predictions = (
    probabilities >= THRESHOLD
).astype(int)

metrics = {
    "model": MODEL_NAME,
    "accuracy": accuracy_score(y_val, predictions),
    "balanced_accuracy": balanced_accuracy_score(
        y_val, predictions
    ),
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_val, probabilities
    ),
    "average_precision": average_precision_score(
        y_val, probabilities
    ),
}

cm = confusion_matrix(
    y_val, predictions, labels=[0, 1]
)

print("\n========================================")
print(" VALIDIERUNGSERGEBNISSE MODELL 4B (2020)")
print("========================================")

for metric, value in metrics.items():
    if metric != "model":
        print(f"{metric:20}: {value:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTestdaten 2021-2022 wurden nicht verwendet.")


# ==================================================
# 7. MODELL UND ERGEBNISSE SPEICHERN
# ==================================================

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Modellgewichte speichern
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
        "seq_len": SEQ_LEN,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    },
    MODELS / "transformer_4B.pt",
)

# Preprocessing speichern
joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "features": FEATURES,
    },
    MODELS / "transformer_4B_preprocessing.joblib",
)

# Kennzahlen speichern
pd.DataFrame([metrics]).to_csv(
    REPORTS / "4B_validation_metrics.csv",
    index=False,
)

# Trainingsverlauf speichern
pd.DataFrame(training_history).to_csv(
    REPORTS / "4B_training_history.csv",
    index=False,
)

# Vorhersagen speichern
pd.DataFrame(
    {
        "label": y_val,
        "score": probabilities,
        "prediction": predictions,
    },
    index=val_ids,
).to_csv(
    REPORTS / "4B_validation_predictions.csv"
)

# Confusion Matrix speichern
pd.DataFrame(
    cm,
    index=["tatsaechlich_0", "tatsaechlich_1"],
    columns=["vorhergesagt_0", "vorhergesagt_1"],
).to_csv(
    REPORTS / "4B_confusion_matrix.csv"
)

# Trainingsinformationen speichern
run_info = {
    "model": MODEL_NAME,
    "architecture": "Transformer Encoder",
    "variant": "with_lst",
    "status": "provisional",
    "target_day_assumption": config["target_day_assumption"],
    "verification": config["verification"],
    "training_years": "2006-2019",
    "validation_year": 2020,
    "test_used": False,
    "features": FEATURES,
    "feature_count": NUM_FEATURES,
    "sequence_length": SEQ_LEN,
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "threshold": THRESHOLD,
    "d_model": D_MODEL,
    "attention_heads": NHEAD,
    "encoder_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "train_samples": len(y_train),
    "validation_samples": len(y_val),
    "trainable_parameters": num_parameters,
    "torch_version": torch.__version__,
    "sklearn_version": sklearn.__version__,
    "imputation": "Median, fitted on train only",
    "scaling": "StandardScaler, fitted on train only",
}

(REPORTS / "4B_run_info.json").write_text(
    json.dumps(
        run_info,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n========================================")
print(" MODELL 4B ERFOLGREICH GESPEICHERT")
print("========================================")
print("Modell:", MODELS / "transformer_4B.pt")
print("Ergebnisse:", REPORTS)
print("Testdaten wurden nicht verwendet.")
