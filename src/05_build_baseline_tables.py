from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "provisional"
OUT = DATA / "baseline"
OUT.mkdir(parents=True, exist_ok=True)

config = json.loads(
    (DATA / "input_config.json").read_text(encoding="utf-8")
)

features_with_lst = config["features_with_lst"]
features_without_lst = config["features_without_lst"]
last_input_index = config["input_time_indices"][1]

manifest = pd.read_csv(
    ROOT / "data" / "processed" / "sample_split.csv"
)

for split in ["train", "validation", "test"]:
    df = pd.read_csv(
        DATA / f"{split}_sequences.csv",
        parse_dates=["time"],
    )

    # Genau den letzten Eingabetag auswählen.
    # Fehlende Werte bleiben fehlend.
    last = df.loc[
        df["time_idx"].eq(last_input_index)
    ].copy()

    if last["sample_id"].duplicated().any():
        raise ValueError(f"{split}: Doppelte Samples am letzten Tag.")

    expected = manifest.loc[
        manifest["split"].eq(split)
    ].set_index("sample_id").sort_index()

    last = last.set_index("sample_id").sort_index()

    if set(last.index) != set(expected.index):
        raise ValueError(f"{split}: Sample-Zuordnung stimmt nicht.")

    last = last.loc[expected.index]

    if not last["label"].eq(expected["label"]).all():
        raise ValueError(f"{split}: Klassen stimmen nicht überein.")

    if not last["split"].eq(split).all():
        raise ValueError(f"{split}: Falsche Teilmenge enthalten.")

    # Nur ausdrücklich ausgewählte Messgrössen als Eingaben.
    X_with_lst = last[features_with_lst]
    X_without_lst = last[features_without_lst]

    # Zielvariable und Datum separat speichern.
    y = last[["label"]]
    metadata = last[["time"]].rename(
        columns={"time": "last_input_date"}
    )

    X_with_lst.to_csv(OUT / f"{split}_X_with_lst.csv")
    X_without_lst.to_csv(OUT / f"{split}_X_without_lst.csv")
    y.to_csv(OUT / f"{split}_y.csv")
    metadata.to_csv(OUT / f"{split}_metadata.csv")

    print(
        f"{split:12} Samples: {len(last):5} | "
        f"mit LST: {X_with_lst.shape[1]} Merkmale | "
        f"ohne LST: {X_without_lst.shape[1]} Merkmale"
    )

description = {
    "status": "provisional",
    "input_time_index": last_input_index,
    "representation": "Messwerte eines einzelnen Eingabetages",
    "model_1A": "X_without_lst",
    "model_1B": "X_with_lst",
    "model_2A": "X_with_lst",
    "row_identifier": "sample_id ist Index, kein Eingabemerkmal",
    "preprocessing": "Noch keine Imputation oder Skalierung",
    "target_day_assumption": config["target_day_assumption"],
}

(OUT / "baseline_config.json").write_text(
    json.dumps(description, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nGespeichert unter data/processed/provisional/baseline/")
print("Zeitreihen bleiben erhalten.")
print("Fehlende Werte wurden nicht ersetzt.")