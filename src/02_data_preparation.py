
from pathlib import Path
import pandas as pd

# ==========================================
# WildFireForecast - Data Preparation
# Step 1: Filter by year (2015-2022)
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    "positives": {
        "separator": ",",
        "date_format": "%Y-%m-%d"
    },
    "negatives": {
        "separator": ";",
        "date_format": "%d.%m.%y"
    }
}

for name, config in datasets.items():

    print(f"\n--- {name.upper()} ---")

    # 1. Load original data
    df = pd.read_csv(
        RAW_DIR / f"{name}.csv",
        sep=config["separator"],
        low_memory=False
    )

    print("Original rows:", len(df))

    # 2. Convert dates
    df["time"] = pd.to_datetime(
        df["time"],
        format=config["date_format"],
        errors="coerce"
    )

    print("Invalid dates:", df["time"].isna().sum())

    # 3. Filter years 2015-2022
    filtered = df.loc[
        df["time"].between(
            "2015-01-01",
            "2022-12-31"
        )
    ].copy()

    print("Filtered rows:", len(filtered))

    # 4. Check year distribution
    print("\nRows per year:")
    print(
        filtered["time"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    # 5. Save intermediate dataset
    output_file = (
        INTERIM_DIR / f"{name}_2015_2022.csv"
    )

    filtered.to_csv(output_file, index=False)

    print("\nSaved:", output_file)

print("\nData preparation step 1 completed.")
print("Original files remain unchanged.")
