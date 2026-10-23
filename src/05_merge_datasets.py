
from pathlib import Path
import pandas as pd

# ==========================================
# WildFireForecast - Merge Datasets
# Italy 2015-2022
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

# 1. Gefilterte Italien-Daten laden
positives = pd.read_csv(
    INTERIM / "positives_italy_2015_2022.csv",
    low_memory=False
)

negatives = pd.read_csv(
    INTERIM / "negatives_italy_2015_2022.csv",
    low_memory=False
)

print("Positive Fälle:", len(positives))
print("Negative Fälle:", len(negatives))

# 2. Spaltenstrukturen überprüfen
if set(positives.columns) != set(negatives.columns):
    raise ValueError(
        "Die Datensätze haben unterschiedliche Spalten!"
    )

# 3. Zielvariable erstellen
positives["fire_occurred"] = 1
negatives["fire_occurred"] = 0

# 4. Datensätze zusammenführen
df = pd.concat(
    [positives, negatives],
    ignore_index=True
)

# 5. Datum überprüfen
df["time"] = pd.to_datetime(df["time"], errors="coerce")

if df["time"].isna().any():
    raise ValueError("Ungültige Datumswerte gefunden!")

if not df["time"].between("2015-01-01", "2022-12-31").all():
    raise ValueError("Beobachtungen ausserhalb des Zeitraums!")

# 6. Ergebnisse untersuchen
print("\nGesamte Beobachtungen:", len(df))

print("\nVerteilung der Zielvariable:")
print(df["fire_occurred"].value_counts())

print("\nProzentuale Verteilung:")
print(
    df["fire_occurred"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFehlende Werte (Top 10):")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nVollständig doppelte Zeilen:")
print(df.duplicated().sum())

# 7. Gemeinsamen Datensatz speichern
output = INTERIM / "wildfire_italy_2015_2022.csv"
df.to_csv(output, index=False)

print("\nGespeichert:", output)
print("Originaldaten bleiben unverändert.")
