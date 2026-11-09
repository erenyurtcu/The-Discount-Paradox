from pathlib import Path
import pandas as pd
import pyreadstat

path = Path("Dominicks_Dataset/Auxiliary/demo.dta")
if not path.exists():
    raise FileNotFoundError(path)

df, meta = pyreadstat.read_dta(str(path))

print("=== DEMOGRAPHICS STRUCTURE ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n=== STORE-RELATED COLUMNS ===")
print([c for c in df.columns if "store" in c.lower()])

print("\n=== SELECTED DEMOGRAPHIC COLUMNS ===")
terms = ["educ", "income", "density", "urban", "age", "house", "family"]
selected = [
    c for c in df.columns
    if any(t in c.lower() for t in terms)
]
print(selected[:100])

print("\n=== FIRST 20 COLUMN NAMES ===")
print(df.columns[:20].tolist())

print("\n=== FIRST 5 ROWS (FIRST 12 COLUMNS) ===")
print(df.iloc[:5, :12].to_string(index=False))

print("\n=== MISSING VALUES IN SELECTED COLUMNS ===")
if selected:
    print(
        df[selected]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(25)
        .to_string()
    )

print("\nDEMOGRAPHICS SCHEMA INSPECTION COMPLETED")
