
from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path.cwd()
SOURCE = ROOT / "Dominicks_Dataset" / "Auxiliary" / "demo.dta"
OUTPUT = ROOT / "Dominicks_Dataset" / "Auxiliary" / "demographics_clean.parquet"

df, metadata = pyreadstat.read_dta(str(SOURCE))

selected = [
    "store",
    "scluster",
    "zone",
    "weekvol",
    "age9",
    "age60",
    "educ",
    "income",
    "incsigma",
    "gini",
    "hsizeavg",
    "density",
    "urban",
]

missing_columns = [c for c in selected if c not in df.columns]
if missing_columns:
    raise RuntimeError(f"Missing columns: {missing_columns}")

demo = df[selected].copy()

demo["store"] = pd.to_numeric(
    demo["store"], errors="coerce"
)

# Remove records without a usable store identifier.
demo = demo.dropna(subset=["store"])

if not (demo["store"] % 1 == 0).all():
    raise RuntimeError("Non-integer store IDs detected")

demo["store"] = demo["store"].astype("int32")

if demo["store"].duplicated().any():
    duplicates = demo.loc[
        demo["store"].duplicated(keep=False), "store"
    ].tolist()
    raise RuntimeError(f"Duplicate store IDs: {duplicates}")

# Prefix demographic variables to prevent collisions.
demo = demo.rename(
    columns={
        c: f"demo_{c}"
        for c in demo.columns
        if c != "store"
    }
)

demo.to_parquet(
    OUTPUT,
    index=False,
    engine="pyarrow",
    compression="zstd"
)

print("=== DEMOGRAPHICS PREPARATION ===")
print("Rows:", len(demo))
print("Unique stores:", demo["store"].nunique())
print("Columns:", len(demo.columns))
print("\nMissing values:")
print(demo.isna().sum().to_string())
print("\nSaved:", OUTPUT)

assert len(demo) == 107
assert demo["store"].is_unique

print("\nDEMOGRAPHICS PREPARATION PASSED")
