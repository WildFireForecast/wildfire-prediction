
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim"

columns = [
    "sample_id",
    "time_idx",
    "ignition_points",
    "burned_areas",
    "burned_area_has",
]

for name in ["positives", "negatives"]:
    df = pd.read_csv(
        DATA / f"{name}_italy.csv",
        usecols=columns
    )

    df["time_idx"] = pd.to_numeric(
        df["time_idx"], errors="coerce"
    )

    target = df[df["time_idx"] == 59].copy()
    n_samples = df["sample_id"].nunique()

    print(f"\n=== {name.upper()} ===")
    print("Samples insgesamt:", n_samples)
    print("Samples mit Zieltag 59:",
          target["sample_id"].nunique())

    for col in [
        "ignition_points",
        "burned_areas",
        "burned_area_has"
    ]:
        values = pd.to_numeric(
            target[col], errors="coerce"
        )

        present = target.loc[
            values.gt(0), "sample_id"
        ].nunique()

        missing = target.loc[
            values.isna(), "sample_id"
        ].nunique()

        print(f"\n{col}")
        print("Positiv am Zieltag:", present)
        print("Ohne positiven Wert am Zieltag:",
              n_samples - present)
        print("Fehlender Wert am Zieltag:", missing)

print("\n=== PRÜFUNG ABGESCHLOSSEN ===")
