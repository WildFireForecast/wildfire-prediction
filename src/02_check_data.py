from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

for name in ["positives", "negatives"]:
    df = pd.read_csv(DATA / f"{name}_italy.csv")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values(["sample_id", "time_idx"])

    groups = df.groupby("sample_id")
    lengths = groups.size()

    duplicates = df.duplicated(["sample_id", "time_idx"]).sum()
    date_steps = groups["time"].diff()
    irregular_steps = (
        date_steps.notna() & date_steps.ne(pd.Timedelta(days=1))
    ).sum()

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100).round(2),
    }).sort_values("missing_percent", ascending=False)

    missing.to_csv(REPORTS / f"{name}_italy_missing_values.csv")

    # Eine Übersicht pro Sample für spätere Prüfungen.
    samples = groups.agg(
        start_date=("time", "min"),
        end_date=("time", "max"),
        rows=("time", "size"),
        longitude=("x", "first"),
        latitude=("y", "first"),
    )
    samples.to_csv(REPORTS / f"{name}_italy_samples.csv")

    print(f"\n--- {name.upper()}: ITALIEN ---")
    print(f"Samples: {len(samples):,}")
    print(f"Zeilen pro Sample: {lengths.min()} bis {lengths.max()}")
    print(f"Doppelte Sample/Zeitindex-Paare: {duplicates}")
    print(f"Datumsabstände ungleich einem Tag: {irregular_steps}")
    print(f"Zeitraum: {df['time'].min().date()} "
          f"bis {df['time'].max().date()}")

    print("\nSpalten mit fehlenden Werten:")
    affected = missing.loc[missing["missing_count"].gt(0)]
    print(affected.to_string() if not affected.empty else "Keine")

    print("\nSamples nach Endjahr:")
    print(samples["end_date"].dt.year.value_counts().sort_index())

print("\nPrüfberichte gespeichert unter reports/.")
