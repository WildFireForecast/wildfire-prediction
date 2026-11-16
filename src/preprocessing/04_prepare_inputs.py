from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed" / "provisional"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "aspect", "curvature", "d2m", "dem", "lai",
    "lst_day", "lst_night", "ndvi", "rh",
    "roads_distance", "slope", "smi", "sp", "ssrd",
    "t2m", "tp", "wind_direction", "wind_speed",
    "lc_agriculture", "lc_forest", "lc_grassland",
    "lc_settlement", "lc_shrubland",
    "lc_sparse_vegetation", "lc_water_bodies",
    "lc_wetland", "population",
]

manifest = pd.read_csv(
    ROOT / "data" / "processed" / "sample_split.csv"
)

if manifest["sample_id"].duplicated().any():
    raise ValueError("Doppelte Sample-IDs in der Aufteilung.")

parts = []
for name in ["positives", "negatives"]:
    df = pd.read_csv(
        DATA / f"{name}_italy.csv",
        usecols=["sample_id", "time", "time_idx"] + FEATURES,
        parse_dates=["time"],
    )
    parts.append(df)

data = pd.concat(parts, ignore_index=True)

data = data.merge(
    manifest[["sample_id", "label", "split"]],
    on="sample_id",
    how="inner",
    validate="many_to_one",
)

if set(data["sample_id"]) != set(manifest["sample_id"]):
    raise ValueError("Nicht alle vorgesehenen Samples wurden gefunden.")

if data.duplicated(["sample_id", "time_idx"]).any():
    raise ValueError("Doppelte Zeitindizes innerhalb eines Samples.")

indices = data.groupby("sample_id")["time_idx"].agg(
    lambda values: set(values) == set(range(60))
)
if not indices.all():
    raise ValueError("Nicht jedes Sample enthält genau die Indizes 0–59.")

# Vorläufige Annahme: Index 59 ist der Zieltag.
inputs = data.loc[data["time_idx"].between(0, 58)].copy()
inputs = inputs.sort_values(["sample_id", "time_idx"])

steps = inputs.groupby("sample_id")["time"].diff().dropna()
if not steps.eq(pd.Timedelta(days=1)).all():
    raise ValueError("Die Eingabezeitreihen sind nicht täglich lückenlos.")

print("\nVorläufige Eingabedaten:")
for split in ["train", "validation", "test"]:
    subset = inputs.loc[inputs["split"].eq(split)].copy()
    if subset.empty:
        raise ValueError(f"Teilmenge {split} ist leer.")

    subset.to_csv(OUT / f"{split}_sequences.csv", index=False)

    print(
        f"{split:12} "
        f"Samples: {subset['sample_id'].nunique():5} | "
        f"Zeilen: {len(subset):7} | Tage pro Sample: 59"
    )

configuration = {
    "status": "provisional",
    "target_day_assumption": "Letzte Zeile, time_idx 59",
    "verification": "Zeitbezug der 60-Tage-Version noch zu bestätigen",
    "input_time_indices": [0, 58],
    "features_with_lst": FEATURES,
    "features_without_lst": [
        c for c in FEATURES if c not in {"lst_day", "lst_night"}
    ],
    "metadata_not_features": [
        "sample_id", "time", "time_idx", "label", "split"
    ],
    "missing_values": "Unverändert; noch keine Imputation",
}

(OUT / "input_config.json").write_text(
    json.dumps(configuration, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nGespeichert unter data/processed/provisional/")
print("Brandspalten sind nicht enthalten.")
print("Fehlende Werte wurden nicht ersetzt.")