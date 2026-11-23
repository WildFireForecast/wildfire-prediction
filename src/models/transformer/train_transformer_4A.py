
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
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Reproduzierbarkeit
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Projektpfade
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "provisional"
MODELS = ROOT / "models" / "transformer" / "baseline"
REPORTS = ROOT / "reports" / "transformer" / "baseline"

# Modellkonfiguration
MODEL_NAME = "4A"
SEQ_LEN = 59
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# Bestehende Feature-Konfiguration laden
config = json.loads(
    (DATA / "input_config.json").read_text(encoding="utf-8")
)

FEATURES = config["features_without_lst"]
NUM_FEATURES = len(FEATURES)

print(f"Modell: {MODEL_NAME}")
print(f"Anzahl Features: {NUM_FEATURES}")
print(f"Sequenzlänge: {SEQ_LEN} Tage")



# --------------------------------------------------
# Schritt 5: Sequenzdaten laden und überprüfen
# --------------------------------------------------

manifest = pd.read_csv(
    ROOT / "data" / "processed" / "sample_split.csv"
)

def load_sequence_data(split, features):
    """Lädt und überprüft die 59-Tage-Sequenzen."""

    if split not in {"train", "validation"}:
        raise ValueError(
            "Für die Modellentwicklung sind nur "
            "Training und Validierung erlaubt."
        )

    df = pd.read_csv(DATA / f"{split}_sequences.csv")

    expected = (
        manifest.loc[manifest["split"].eq(split)]
        .set_index("sample_id")
        .sort_index()
    )

    # Sample-Zuordnung überprüfen
    if df["sample_id"].duplicated().all():
        raise ValueError("Ungültige Sample-Struktur.")

    if set(df["sample_id"]) != set(expected.index):
        raise ValueError(
            f"{split}: Sample-IDs stimmen nicht überein."
        )

    # Reihenfolge herstellen
    df = df.sort_values(["sample_id", "time_idx"])
    sample_ids = df["sample_id"].unique()

    if not np.array_equal(sample_ids, expected.index.to_numpy()):
        raise ValueError(
            f"{split}: Reihenfolge der Samples stimmt nicht."
        )

    # Jeder Sample benötigt 59 Zeitschritte
    if len(df) != len(sample_ids) * SEQ_LEN:
        raise ValueError(
            f"{split}: Nicht jedes Sample hat 59 Tage."
        )

    if not df.groupby("sample_id")["time_idx"].apply(
        lambda x: np.array_equal(
            x.to_numpy(), np.arange(SEQ_LEN)
        )
    ).all():
        raise ValueError(
            f"{split}: Zeitindizes sind unvollständig."
        )

    X = df[features].to_numpy(dtype=np.float32)
    y = expected["label"].to_numpy(dtype=np.int64)

    if np.isinf(X).any():
        raise ValueError(
            f"{split}: Unendliche Werte gefunden."
        )

    return X, y, expected.index


# Trainingsdaten laden
X_train, y_train, train_ids = load_sequence_data(
    "train", FEATURES
)

# Validierungsdaten laden
X_val, y_val, val_ids = load_sequence_data(
    "validation", FEATURES
)

# Überschneidungen verhindern
if not train_ids.intersection(val_ids).empty:
    raise ValueError(
        "Data Leakage: Samples in beiden Datengruppen!"
    )

print("\nDaten erfolgreich geladen!")
print(f"Trainingssamples: {len(y_train)}")
print(f"Validierungssamples: {len(y_val)}")
print(f"Anzahl Features: {NUM_FEATURES}")
print(f"Zeitschritte pro Sample: {SEQ_LEN}")
print("Keine Überschneidung der Sample-IDs.")
print("Testdaten wurden nicht geladen.")



# --------------------------------------------------
# Schritt 6: Imputation und Standardisierung
# --------------------------------------------------

# Fehlende Werte nur anhand der Trainingsdaten ersetzen
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Standardisierung nur anhand der Trainingsdaten
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)

# Überprüfung der Feature-Anzahl
if X_train_scaled.shape[1] != NUM_FEATURES:
    raise ValueError("Feature-Anzahl stimmt nicht.")

# Überprüfung auf NaN und unendliche Werte
if not np.isfinite(X_train_scaled).all():
    raise ValueError("Ungültige Werte in Trainingsdaten.")

if not np.isfinite(X_val_scaled).all():
    raise ValueError("Ungültige Werte in Validierungsdaten.")

# Umformen in 3D-Sequenzen
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



# --------------------------------------------------
# Schritt 7: Transformer-Architektur
# --------------------------------------------------

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

        # Projektion der 25 Features auf 64 Dimensionen
        self.input_projection = nn.Linear(
            num_features, d_model
        )

        # Lernbare Positionsinformationen für 59 Tage
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, SEQ_LEN, d_model)
        )

        nn.init.normal_(
            self.pos_embedding, std=0.02
        )

        # Transformer Encoder
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

        # Klassifikationsschicht
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):

        # x: (Batch, 59, 25)

        x = self.input_projection(x)

        # Positionsinformationen hinzufügen
        x = x + self.pos_embedding

        # Self-Attention über die Zeitreihe
        x = self.transformer(x)

        # Informationen aller Tage zusammenfassen
        x = x.mean(dim=1)

        # Waldbrand-Logit berechnen
        logits = self.classifier(x)

        return logits.squeeze(-1)


# Modell 4A initialisieren
model = WildfireTransformer(
    num_features=NUM_FEATURES
)

print("\nTransformer 4A erfolgreich erstellt!")
print(
    "Trainierbare Parameter:",
    sum(p.numel() for p in model.parameters())
)

# Architektur mit Beispieldaten prüfen
dummy_input = torch.zeros(
    2, SEQ_LEN, NUM_FEATURES
)

with torch.no_grad():
    dummy_output = model(dummy_input)

assert dummy_output.shape == (2,)

print("Eingabeform:", tuple(dummy_input.shape))
print("Ausgabeform:", tuple(dummy_output.shape))
print("Architekturprüfung erfolgreich!")



# --------------------------------------------------
# Schritt 8: Transformer 4A trainieren
# --------------------------------------------------

# Trainingsdaten in PyTorch-Tensoren umwandeln
train_dataset = TensorDataset(
    torch.tensor(X_train_3d, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Verlustfunktion für binäre Klassifikation
criterion = nn.BCEWithLogitsLoss()

# Optimierungsalgorithmus
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# Trainingsverlauf speichern
training_history = []

print("\n========================================")
print(" Starte Training: Transformer Modell 4A")
print("========================================")

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        # Vorhersagen berechnen
        logits = model(batch_x)

        # Trainingsverlust berechnen
        loss = criterion(logits, batch_y)

        # Backpropagation
        loss.backward()

        # Modellgewichte aktualisieren
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

print("\nTraining von Modell 4A abgeschlossen!")



# --------------------------------------------------
# Schritt 9: Validierung von Modell 4A
# --------------------------------------------------

model.eval()

# Validierungsdaten in Tensoren umwandeln
val_tensor = torch.tensor(
    X_val_3d, dtype=torch.float32
)

# Vorhersagen ohne Gradientenberechnung
with torch.no_grad():
    val_logits = model(val_tensor)
    probabilities = torch.sigmoid(
        val_logits
    ).cpu().numpy()

# Klassifikation mit Schwellenwert 0.5
predictions = (probabilities >= 0.5).astype(int)

# Bewertungskennzahlen berechnen
metrics = {
    "model": "4A",
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

# Confusion Matrix
cm = confusion_matrix(
    y_val, predictions, labels=[0, 1]
)

print("\n========================================")
print(" VALIDIERUNGSERGEBNISSE MODELL 4A (2020)")
print("========================================")

for metric, value in metrics.items():
    if metric != "model":
        print(f"{metric:20}: {value:.4f}")

print("\nConfusion Matrix:")
print(cm)
print("\nTestdaten 2021-2022 wurden nicht verwendet.")



# --------------------------------------------------
# Schritt 10: Modell und Ergebnisse speichern
# --------------------------------------------------

# Ausgabeordner erstellen
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# 1. Transformer-Modell speichern
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
        "seq_len": SEQ_LEN,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.2,
    },
    MODELS / "transformer_4A.pt",
)

# 2. Imputer und Scaler speichern
joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "features": FEATURES,
    },
    MODELS / "transformer_4A_preprocessing.joblib",
)

# 3. Validierungskennzahlen speichern
pd.DataFrame([metrics]).to_csv(
    REPORTS / "validation_metrics.csv",
    index=False,
)

# 4. Trainingsverlauf speichern
pd.DataFrame(training_history).to_csv(
    REPORTS / "4A_training_history.csv",
    index=False,
)

# 5. Einzelne Vorhersagen speichern
pd.DataFrame(
    {
        "label": y_val,
        "score": probabilities,
        "prediction": predictions,
    },
    index=val_ids,
).to_csv(
    REPORTS / "4A_validation_predictions.csv"
)

# 6. Confusion Matrix speichern
pd.DataFrame(
    cm,
    index=["tatsaechlich_0", "tatsaechlich_1"],
    columns=["vorhergesagt_0", "vorhergesagt_1"],
).to_csv(
    REPORTS / "4A_confusion_matrix.csv"
)

# 7. Informationen zum Trainingslauf speichern
run_info = {
    "model": "4A",
    "architecture": "Transformer Encoder",
    "status": "provisional",
    "target_day_assumption": config["target_day_assumption"],
    "verification": config["verification"],
    "training_years": "2006-2019",
    "validation_year": 2020,
    "test_used": False,
    "features": FEATURES,
    "sequence_length": SEQ_LEN,
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "threshold": 0.5,
    "d_model": 64,
    "attention_heads": 4,
    "encoder_layers": 2,
    "dropout": 0.2,
    "train_samples": len(y_train),
    "validation_samples": len(y_val),
}

(REPORTS / "run_info.json").write_text(
    json.dumps(
        run_info,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("\n========================================")
print(" MODELL 4A ERFOLGREICH GESPEICHERT")
print("========================================")
print("Modell:", MODELS / "transformer_4A.pt")
print("Ergebnisse:", REPORTS)
print("Testdaten wurden nicht verwendet.")

