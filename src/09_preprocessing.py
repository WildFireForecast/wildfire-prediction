
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer

# ==============================================
# WildFireForecast - Step 09: Preprocessing
# Baseline Model A
# ==============================================

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# 1. Input features for Model A
FEATURES = [
    # Weather
    "t2m",
    "d2m",
    "rh",
    "tp",
    "wind_speed",
    "wind_direction",
    "sp",
    "ssrd",

    # Vegetation
    "ndvi",
    "lai",
    "smi",

    # Topography
    "dem",
    "slope",
    "aspect",
    "curvature",

    # Land cover
    "lc_forest",
    "lc_grassland",
    "lc_shrubland",
    "lc_agriculture",
    "lc_sparse_vegetation",
    "lc_settlement",
    "lc_water_bodies",
    "lc_wetland",

    # Human influence
    "roads_distance",
    "population"
]

TARGET = "fire_occurred"

# 2. Load temporal splits
files = {
    "train": "train_2015_2020.csv",
    "validation": "validation_2021.csv",
    "test": "test_2022.csv"
}

datasets = {}
features = {}

for name, filename in files.items():

    df = pd.read_csv(
        INTERIM / filename,
        low_memory=False
    )

    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {name}: {missing_columns}"
        )

    datasets[name] = df

    # Convert all selected features to numbers
    X = df[FEATURES].apply(
        pd.to_numeric,
        errors="raise"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    features[name] = X

    print(f"\n===== {name.upper()} =====")
    print("Observations:", len(df))
    print("Features:", len(FEATURES))
    print("Missing values before imputation:")
    print(
        X.isna().sum()[
            X.isna().sum() > 0
        ]
    )

# 3. Fit imputer ONLY on training data
imputer = SimpleImputer(
    strategy="median"
)

imputer.fit(features["train"])

# Show calculated LAI median
lai_index = FEATURES.index("lai")
lai_median = imputer.statistics_[lai_index]

print("\n===== IMPUTATION =====")
print("LAI median from training:", lai_median)

# 4. Apply the same imputer to all splits
for name, X in features.items():

    transformed = imputer.transform(X)

    X_processed = pd.DataFrame(
        transformed,
        columns=FEATURES
    )

    # Preserve target labels
    X_processed[TARGET] = (
        datasets[name][TARGET]
        .astype(int)
        .to_numpy()
    )

    # Check for remaining missing values
    remaining = X_processed.isna().sum().sum()

    if remaining > 0:
        raise ValueError(
            f"{name}: {remaining} missing values remain"
        )

    # Save processed dataset
    output = PROCESSED / f"{name}_processed.csv"

    X_processed.to_csv(
        output,
        index=False
    )

    print(f"\n{name.upper()} processed")
    print("Shape:", X_processed.shape)
    print("Remaining missing values:", remaining)
    print("Saved:", output)

# 5. Save fitted imputer and feature names
joblib.dump(
    {
        "imputer": imputer,
        "features": FEATURES
    },
    MODELS / "preprocessing_model_a.joblib"
)

print("\nPreprocessing completed successfully!")
print("Original datasets remain unchanged.")
