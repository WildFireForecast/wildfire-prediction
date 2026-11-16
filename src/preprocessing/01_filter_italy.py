from pathlib import Path
from urllib.request import urlopen
import ssl
import shutil

import certifi
import geopandas as gpd
import pandas as pd


# Projektordner unabhängig vom aktuellen Terminal-Verzeichnis finden.
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim"
BORDERS = RAW / "boundaries"

OUT.mkdir(parents=True, exist_ok=True)
BORDERS.mkdir(parents=True, exist_ok=True)

# Eurostat/GISCO: Ländergrenzen 2024, Massstab 1:1 Mio.,
# Koordinatensystem WGS84 (EPSG:4326).
URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "countries/geojson/CNTR_RG_01M_2024_4326.geojson"
)
boundary_file = BORDERS / "countries_2024_4326.geojson"

if not boundary_file.exists():
    print("Lade Landesgrenzen von Eurostat/GISCO ...", flush=True)
    temporary = boundary_file.with_suffix(".download")
    context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(URL, context=context, timeout=120) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    temporary.replace(boundary_file)

countries = gpd.read_file(boundary_file)
italy = countries.loc[countries["CNTR_ID"].eq("IT")].copy()

if italy.empty:
    raise ValueError("Italien wurde in der Grenzdatei nicht gefunden.")

italy = italy.to_crs("EPSG:4326")
italy_shape = italy.geometry.union_all()

if italy_shape.is_empty or not italy_shape.is_valid:
    raise ValueError("Die Italien-Geometrie ist leer oder ungültig.")

(BORDERS / "source.txt").write_text(
    f"Quelle: Eurostat/GISCO, Countries 2024\n"
    f"URL: {URL}\n"
    "Massstab: 1:1 Million\n"
    "CRS: EPSG:4326\n"
    "Filter: Standort liegt innerhalb oder auf der Grenze Italiens.\n",
    encoding="utf-8",
)

summary = []

for name in ["positives", "negatives"]:
    print(f"\nLese {name}.csv ...", flush=True)
    df = pd.read_csv(RAW / f"{name}.csv")

    required = ["sample", "time_idx", "x", "y"]
    if df[required].isna().any().any():
        raise ValueError(f"{name}: Fehlende IDs, Zeitindizes oder Koordinaten.")

    groups = df.groupby("sample")
    if not groups.size().eq(60).all():
        raise ValueError(f"{name}: Nicht jedes Sample hat 60 Zeilen.")

    if groups[["x", "y"]].nunique().gt(1).any().any():
        raise ValueError(f"{name}: Koordinaten wechseln innerhalb eines Samples.")

    # Pro Sample genügt ein Standort für die geografische Prüfung.
    locations = df[["sample", "x", "y"]].drop_duplicates("sample")

    points = gpd.GeoSeries(
        gpd.points_from_xy(locations["x"], locations["y"]),
        index=locations.index,
        crs="EPSG:4326",
    )

    in_italy = points.intersects(italy_shape)
    selected_ids = locations.loc[in_italy, "sample"]

    if selected_ids.empty:
        raise ValueError(f"{name}: Kein Sample in Italien gefunden.")

    result = df.loc[df["sample"].isin(selected_ids)].copy()

    # Eindeutige Kennung über beide Quelldateien hinweg.
    # Die ursprüngliche Spalte 'sample' bleibt erhalten.
    result["sample_id"] = name + "_" + result["sample"].astype(str)
    result["source_file"] = f"{name}.csv"

    result = result.sort_values(["sample", "time_idx"])

    destination = OUT / f"{name}_italy.csv"
    temporary = destination.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(destination)

    total = df["sample"].nunique()
    selected = result["sample"].nunique()

    print(f"Samples insgesamt: {total:,}")
    print(f"Samples in Italien: {selected:,}")
    print(f"Zeilen in Italien: {len(result):,}")
    print(f"Gespeichert: {destination.name}")

    summary.append({
        "source_file": f"{name}.csv",
        "samples_total": total,
        "samples_italy": selected,
        "rows_italy": len(result),
    })

pd.DataFrame(summary).to_csv(
    OUT / "italy_filter_summary.csv", index=False
)

print("\nFertig. Die beiden Original-CSV-Dateien bleiben unverändert.")
