
from pathlib import Path
import pandas as pd

# ==========================================
# WildFireForecast – Data Quality Check
# Italy 2015–2022
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"

df = pd.read_csv(
    DATA / "wildfire_italy_2015_2022.csv",
    parse_dates=["time"],
    low_memory=False
)

# 1. Allgemeine Informationen
print("\n===== DATASET OVERVIEW =====")
print("Observations:", len(df))
print("Features:", len(df.columns))

print("\nTarget distribution:")
print(df["fire_occurred"].value_counts())

print("\nTarget distribution (%):")
print(
    df["fire_occurred"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 2. Missing values
print("\n===== MISSING VALUES =====")

missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Percentage": (
        df.isna().mean() * 100
    ).round(2)
})

missing = missing.sort_values(
    "Percentage",
    ascending=False
)

print(missing[missing["Missing"] > 0])

# 3. Missing values by target class
print("\n===== MISSING VALUES BY CLASS =====")

for target in [0, 1]:

    subset = df[df["fire_occurred"] == target]

    print(f"\nClass {target} ({len(subset)} observations)")

    missing_pct = (
        subset.isna().mean() * 100
    ).sort_values(ascending=False)

    print(missing_pct.head(10).round(2))

# 4. Duplicates
print("\n===== DUPLICATES =====")

print("Exact duplicates:", df.duplicated().sum())

# Same location and date
keys = ["time", "x", "y"]

groups = df.groupby(
    keys,
    dropna=False
).agg(
    observations=("fire_occurred", "size"),
    classes=("fire_occurred", "nunique")
)

print(
    "Repeated location/date combinations:",
    (groups["observations"] > 1).sum()
)

print(
    "Conflicting labels at same location/date:",
    (groups["classes"] > 1).sum()
)

# 5. Check potential data leakage
print("\n===== POTENTIAL DATA LEAKAGE =====")

potential_leakage = [
    "burned_areas",
    "burned_area_has",
    "ignition_points",
    "sample",
    "time_idx"
]

for column in potential_leakage:
    if column in df.columns:

        print(f"\nVariable: {column}")

        print(
            df.groupby("fire_occurred")[column]
            .agg(["count", "nunique"])
        )

print("\nData quality check completed.")
print("Original files remain unchanged.")
