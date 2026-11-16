from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

parts = []

for name, label in [("positives", 1), ("negatives", 0)]:
    df = pd.read_csv(
        DATA / f"{name}_italy.csv",
        usecols=["sample_id", "time", "x", "y"],
        parse_dates=["time"],
    )

    samples = df.groupby("sample_id").agg(
        start_date=("time", "min"),
        end_date=("time", "max"),
        rows=("time", "size"),
        longitude=("x", "first"),
        latitude=("y", "first"),
    ).reset_index()

    samples["label"] = label
    parts.append(samples)

manifest = pd.concat(parts, ignore_index=True)

if manifest["sample_id"].duplicated().any():
    raise ValueError("Sample-IDs sind nicht eindeutig.")

if not manifest["rows"].eq(60).all():
    raise ValueError("Nicht jedes Sample enthält 60 Zeilen.")

years = manifest["end_date"].dt.year

if not years.between(2006, 2022).all():
    raise ValueError("Unerwartetes Endjahr im Datensatz.")

manifest["split"] = "train"
manifest.loc[years.eq(2020), "split"] = "validation"
manifest.loc[years.ge(2021), "split"] = "test"

# Prüfen, ob Zeitreihen über die Grenzen der Teilmengen reichen.
crossing = (
    (
        manifest["split"].eq("validation")
        & manifest["start_date"].lt(pd.Timestamp("2020-01-01"))
    )
    | (
        manifest["split"].eq("test")
        & manifest["start_date"].lt(pd.Timestamp("2021-01-01"))
    )
)

# Grenzüberschreitende Zeitreihen separat dokumentieren.
excluded = manifest.loc[crossing].copy()
excluded["exclusion_reason"] = "Zeitreihe überschreitet Split-Grenze"

excluded.to_csv(
    OUT / "samples_excluded_split_boundary.csv",
    index=False,
)

print(f"\nSamples vor der zeitlichen Bereinigung: {len(manifest)}")
print(f"Ausgeschlossene Samples: {len(excluded)}")

if not excluded.empty:
    print("\nAusschlüsse nach Teilmenge und Klasse:")
    print(
        excluded.groupby(["split", "label"])
        .size()
        .rename("anzahl")
        .to_string()
    )

# Immer vollständige Samples ausschliessen.
manifest = manifest.loc[~crossing].copy()

if manifest.empty:
    raise ValueError("Nach der Bereinigung sind keine Samples übrig.")

manifest = manifest.sort_values(["end_date", "sample_id"])
manifest.to_csv(OUT / "sample_split.csv", index=False)

counts = pd.crosstab(manifest["split"], manifest["label"])
counts = counts.reindex(
    index=["train", "validation", "test"],
    columns=[0, 1],
    fill_value=0,
)
counts.columns = ["negativ", "positiv"]
counts["gesamt"] = counts.sum(axis=1)

print("\nAufteilung der Samples:")
print(counts.to_string())
print(f"\nSamples insgesamt: {len(manifest)}")
print("Gespeichert: data/processed/sample_split.csv")