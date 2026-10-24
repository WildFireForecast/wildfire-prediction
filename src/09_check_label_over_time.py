
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"

for name in ["positives", "negatives"]:
    df = pd.read_csv(
        DATA / f"{name}_italy.csv",
        usecols=["sample_id", "time_idx", "burned_area_has"]
    )

    df["burned_area_has"] = pd.to_numeric(
        df["burned_area_has"], errors="coerce"
    )

    counts = df.groupby("sample_id")[
        "burned_area_has"
    ].nunique(dropna=False)

    print(f"\n=== {name.upper()} ===")
    print("Samples:", len(counts))
    print("Konstant über alle Tage:", (counts == 1).sum())
    print("Wert verändert sich:", (counts > 1).sum())

    for day in [0, 58, 59]:
        target = df[df["time_idx"] == day]
        print(
            f"Tag {day}:",
            (target["burned_area_has"] > 0).sum(),
            "positive Werte"
        )

print("\n=== ZEITLICHE PRÜFUNG ABGESCHLOSSEN ===")
