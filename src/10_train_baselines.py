
from pathlib import Path
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================
# WildFireForecast - Step 10
# Baseline Model Comparison
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# 1. Load training and validation data
train = pd.read_csv(
    DATA_DIR / "train_processed.csv"
)

validation = pd.read_csv(
    DATA_DIR / "validation_processed.csv"
)

TARGET = "fire_occurred"

FEATURES = [
    col for col in train.columns
    if col != TARGET
]

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

print("Training observations:", len(X_train))
print("Validation observations:", len(X_val))
print("Features:", len(FEATURES))

# 2. Define models
models = {
    "Dummy Classifier": DummyClassifier(
        strategy="most_frequent"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# 3. Train and evaluate
for name, model in models.items():

    print(f"\n===== {name.upper()} =====")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    print(
        "Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(
                y_val, predictions
            ), 4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_val, predictions,
                zero_division=0
            ), 4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_val, predictions,
                zero_division=0
            ), 4
        )
    )

    print(
        "F1-Score:",
        round(
            f1_score(
                y_val, predictions,
                zero_division=0
            ), 4
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_val, probabilities
            ), 4
        )
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

print("\nBaseline comparison completed!")
print("Test data from 2022 was not used.")
