
from pathlib import Path
import pandas as pd

# =============================================
# WildFireForecast - Step 08: Temporal Split
# Italy 2015-2022
# =============================================

# 1. Projektpfade definieren
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "interim"

# 2. Zusammengeführten Datensatz einlesen
df = pd.read_csv(
    DATA_DIR / "wildfire_italy_2015_2022.csv",
    parse_dates=["time"],
    low_memory=False
)

print("===== ORIGINAL DATASET =====")
print("Total observations:", len(df))
print("Total columns:", len(df.columns))

# 3. Daten überprüfen
if df["time"].isna().any():
    raise ValueError("Missing or invalid dates found!")

if not df["fire_occurred"].isin([0, 1]).all():
    raise ValueError("Invalid target values found!")

if not df["time"].between(
    "2015-01-01", "2022-12-31"
).all():
    raise ValueError("Dates outside 2015-2022 found!")

# 4. Jahr aus Datum extrahieren
df["year"] = df["time"].dt.year

# 5. Zeitliche Aufteilung
train = df.loc[
    df["year"].between(2015, 2020)
].copy()

validation = df.loc[
    df["year"] == 2021
].copy()

test = df.loc[
    df["year"] == 2022
].copy()

# 6. Überprüfen, ob alle Zeilen erhalten bleiben
assert (
    len(train) + len(validation) + len(test)
    == len(df)
)

splits = {
    "train_2015_2020": train,
    "validation_2021": validation,
    "test_2022": test
}

# 7. Jeden Datensatz untersuchen und speichern
for name, subset in splits.items():

    print(f"\n===== {name.upper()} =====")

    print("Observations:", len(subset))

    print("\nClass distribution:")
    print(subset["fire_occurred"].value_counts())

    print("\nClass distribution (%):")
    print(
        subset["fire_occurred"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMissing values:")
    print(
        subset.isna()
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\nDate range:")
    print(subset["time"].min())
    print(subset["time"].max())

    # Datei speichern
    output = DATA_DIR / f"{name}.csv"
    subset.to_csv(output, index=False)

    print("\nSaved:", output)

print("\nTemporal Split completed successfully!")
print("Original datasets remain unchanged.")
print("No missing values have been imputed yet.")
