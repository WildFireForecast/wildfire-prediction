
from pathlib import Path
import pandas as pd
import geopandas as gpd

# ==========================================
# WildFireForecast - Geographic Filtering
# Italy, 2015-2022
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REFERENCE = ROOT / "data" / "reference"

# 1. Load Italian geographic boundaries
italy = gpd.read_file(
    REFERENCE / "IT-region-01m-4326-2024.geojson"
)

# 2. Standardize coordinate system
if italy.crs is None:
    raise ValueError("Missing coordinate reference system")

italy = italy.to_crs("EPSG:4326")

# 3. Combine all Italian regions
italy_geometry = italy.geometry.union_all()

# 4. Get geographical bounding box
min_x, min_y, max_x, max_y = italy_geometry.bounds

# 5. Process positive and negative datasets
for name in ["positives", "negatives"]:

    print(f"\n===== {name.upper()} =====")

    file_path = INTERIM / f"{name}_2015_2022.csv"
    df = pd.read_csv(file_path, low_memory=False)

    print("Before filtering:", len(df))

    # Convert coordinates into numeric values
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Fast geographic pre-filter
    candidates = df.loc[
        df["x"].between(min_x, max_x)
        & df["y"].between(min_y, max_y)
    ].copy()

    # Convert coordinates into geographic points
    points = gpd.GeoSeries(
        gpd.points_from_xy(
            candidates["x"],
            candidates["y"]
        ),
        index=candidates.index,
        crs="EPSG:4326"
    )

    # Check which points intersect Italy
    inside_italy = points.intersects(italy_geometry)

    filtered = candidates.loc[inside_italy].copy()

    print("After Italy filtering:", len(filtered))

    # Check year distribution
    filtered["time"] = pd.to_datetime(
        filtered["time"],
        errors="coerce"
    )

    print("\nObservations per year:")
    print(
        filtered["time"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    # Save without modifying raw datasets
    output = INTERIM / f"{name}_italy_2015_2022.csv"

    filtered.to_csv(output, index=False)

    print("\nSaved:", output)

print("\nGeographic filtering completed!")
print("Original data remains unchanged.")
