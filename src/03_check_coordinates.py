
from pathlib import Path
import pandas as pd

# Projektverzeichnis
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "interim"

# Beide zeitlich gefilterten Datensätze prüfen
for name in ["positives", "negatives"]:

    file = DATA_DIR / f"{name}_2015_2022.csv"

    df = pd.read_csv(
        file,
        usecols=["x", "y"]
    )

    print(f"\n===== {name.upper()} =====")

    print("\nAnzahl Beobachtungen:")
    print(len(df))

    print("\nErste 5 Koordinaten:")
    print(df.head().to_string(index=False))

    print("\nKoordinatenbereich:")
    print(df[["x", "y"]].agg(["min", "max"]))

    print("\nFehlende Koordinaten:")
    print(df[["x", "y"]].isna().sum())
