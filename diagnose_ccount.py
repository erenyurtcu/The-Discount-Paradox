
from pathlib import Path
import pandas as pd
import pyreadstat

path = Path("Dominicks_Dataset/Auxiliary/ccount.dta")

df, meta = pyreadstat.read_dta(
    str(path),
    usecols=["store", "date", "week", "custcoun"],
    encoding="latin1"
)

print("\n=== VERİ TİPLERİ ===")
print(df.dtypes)

print("\n=== DATE ÖRNEKLERİ ===")
print(df["date"].value_counts(dropna=False).head(15))

print("\n=== WEEK DAĞILIMI ===")
print(df["week"].describe())
print("Negatif hafta:", (df["week"] < 0).sum())
print("Eksik hafta:", df["week"].isna().sum())

print("\n=== STORE DAĞILIMI ===")
print(df["store"].value_counts(dropna=False).head(15))

print("\n=== STORE=0 ÖRNEKLERİ ===")
print(df.loc[df["store"] == 0].head(15).to_string(index=False))

print("\n=== NORMAL WEEK ÖRNEKLERİ ===")
print(
    df.loc[df["week"].between(1, 400)]
      .head(15)
      .to_string(index=False)
)

print("\n=== AYNI STORE-WEEK ÖRNEĞİ ===")
valid = df.dropna(subset=["store", "week"])
duplicates = valid[
    valid.duplicated(["store", "week"], keep=False)
]

print(duplicates.head(20).to_string(index=False))

print("\n=== DATE + STORE + WEEK TEKRARI ===")
print(
    df.duplicated(["store", "date", "week"]).sum()
)

print("\n=== WEEK BAZINDA KAYIT SAYILARI ===")
print(df["week"].value_counts().head(15))

print("\n=== DEĞİŞKEN AÇIKLAMALARI ===")
for col in ["store", "date", "week", "custcoun"]:
    idx = meta.column_names.index(col)
    print(col, ":", meta.column_labels[idx])
