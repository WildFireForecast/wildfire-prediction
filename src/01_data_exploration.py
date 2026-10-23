from pathlib import Path
import pandas as pd

# 1. Speicherort der Rohdaten
data_path = Path(__file__).resolve().parents[1] / "data" / "raw"

# 2. Datensätze einlesen
# Für den ersten Test jeweils nur 10'000 Zeilen
positives = pd.read_csv(
    data_path / "positives.csv",
    sep=",",
    nrows=10000
)

negatives = pd.read_csv(
    data_path / "negatives.csv",
    sep=";",
    nrows=10000
)

# 3. Daten untersuchen
for name, df in [
    ("Positive Fälle", positives),
    ("Negative Fälle", negatives)
]:
    print(f"\n========== {name} ==========")

    # Anzahl Zeilen und Spalten
    print("\nDatensatzgrösse:")
    print(df.shape)

    # Spaltennamen
    print("\nSpalten:")
    print(df.columns.tolist())

    # Erste 5 Zeilen
    print("\nErste 5 Zeilen:")
    print(df.head())

    # Datentypen
    print("\nDatentypen:")
    print(df.dtypes)

    # Fehlende Werte
    missing = pd.DataFrame({
        "Fehlende Werte": df.isnull().sum(),
        "Anteil (%)": (df.isnull().mean() * 100).round(2)
    })

    print("\nFehlende Werte:")
    print(missing[missing["Fehlende Werte"] > 0])