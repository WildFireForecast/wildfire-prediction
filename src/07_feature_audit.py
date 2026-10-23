
from pathlib import Path
import pandas as pd

# ==============================================
# WildFireForecast - Feature Audit
# ==============================================

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT / "data" / "interim"
    / "wildfire_italy_2015_2022.csv"
)

df = pd.read_csv(
    DATA,
    parse_dates=["time"],
    low_memory=False
)

# 1. Positive und negative Fälle pro Jahr
print("\n===== CLASS DISTRIBUTION BY YEAR =====")

yearly = pd.crosstab(
    df["time"].dt.year,
    df["fire_occurred"]
)

yearly.columns = ["No Fire", "Fire"]
yearly["Total"] = yearly.sum(axis=1)

print(yearly)

# 2. Verdächtige Variablen untersuchen
print("\n===== POTENTIAL DATA LEAKAGE =====")

suspicious_features = [
    "burned_areas",
    "burned_area_has",
    "ignition_points"
]

for feature in suspicious_features:

    if feature not in df.columns:
        continue

    print(f"\n--- {feature} ---")

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    print("Unique values:")
    print(
        df.groupby("fire_occurred")[feature]
        .nunique()
    )

    print("\nPercentage of non-zero values:")

    non_zero = values.ne(0) & values.notna()

    result = (
        non_zero
        .groupby(df["fire_occurred"])
        .mean()
        .mul(100)
        .round(2)
    )

    print(result)

# 3. Missing values by year
print("\n===== MISSING VALUES BY YEAR =====")

features = [
    "lst_day",
    "lst_night",
    "lai"
]

for feature in features:

    if feature in df.columns:

        print(f"\n{feature}:")

        missing = (
            df[feature].isna()
            .groupby(df["time"].dt.year)
            .mean()
            .mul(100)
            .round(2)
        )

        print(missing)

print("\nFeature audit completed.")
print("Original data unchanged.")
