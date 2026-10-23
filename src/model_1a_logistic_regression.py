
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# WildFireForecast
# Model 1A: Logistic Regression without LST
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

TARGET = "fire_occurred"

# 1. Load prepared datasets
train = pd.read_csv(
    DATA_DIR / "train_processed.csv"
)

validation = pd.read_csv(
    DATA_DIR / "validation_processed.csv"
)

# 2. Select model features
FEATURES = [
    col for col in train.columns
    if col != TARGET
]

# Check that Model 1A uses exactly 25 features
assert len(FEATURES) == 25
assert "lst_day" not in FEATURES
assert "lst_night" not in FEATURES

# Ensure validation uses the same features
assert all(col in validation.columns for col in FEATURES)

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

print("\n===== MODEL 1A =====")
print("Algorithm: Logistic Regression")
print("Features:", len(FEATURES))
print("Training observations:", len(X_train))
print("Validation observations:", len(X_val))

# 3. Create logistic regression model
model_1a = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

# 4. Train the model
model_1a.fit(X_train, y_train)

# 5. Generate predictions
predictions = model_1a.predict(X_val)

probabilities = model_1a.predict_proba(
    X_val
)[:, 1]

# 6. Evaluate the model
print("\n===== MODEL 1A EVALUATION =====")

print(
    "Accuracy:",
    round(accuracy_score(y_val, predictions), 4)
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_val, predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_val, predictions), 4)
)

print(
    "Recall:",
    round(recall_score(y_val, predictions), 4)
)

print(
    "F1-Score:",
    round(f1_score(y_val, predictions), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_val, probabilities), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

print("\nModel 1A completed successfully!")
print("Test data from 2022 was not used.")
