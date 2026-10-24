
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data/interim/positives_italy.csv"

columns = [
    "sample_id",
    "time_idx",
    "time",
    "burned_areas",
    "ignition_points",
    "burned_area_has",
]

df = pd.read_csv(FILE, usecols=columns)
df["time_idx"] = pd.to_numeric(df["time_idx"])

print("=== PRÜFUNG DER ZEITREIHEN ===")
print("Anzahl Samples:", df["sample_id"].nunique())
print("Anzahl Zeilen:", len(df))
print("Zeitindizes:", sorted(df["time_idx"].unique()))

for column in [
    "burned_areas",
    "ignition_points",
    "burned_area_has",
]:
    values = pd.to_numeric(df[column], errors="coerce")

    fire_rows = df.loc[
        values.fillna(0).gt(0),
        ["sample_id", "time_idx"]
    ]

    first_occurrence = (
        fire_rows.groupby("sample_id")["time_idx"].min()
    )

    print(f"\n=== {column} ===")
    print("Samples mit Brandinformation:",
          len(first_occurrence))

    print("Erstmals bei Index 59:",
          (first_occurrence == 59).sum())

    print("Erstmals vor Index 59:",
          (first_occurrence < 59).sum())

    print("Verteilung der ersten Indizes:")
    print(first_occurrence.value_counts().sort_index().tail(15))

print("\n=== BEISPIEL EINER ZEITREIHE ===")
sample_id = df["sample_id"].iloc[0]
sample = df[df["sample_id"] == sample_id]

print(
    sample[
        ["time_idx", "time",
         "burned_areas", "ignition_points"]
    ].tail(10).to_string(index=False)
)
