
from pathlib import Path
import zipfile
import io
import pandas as pd
import numpy as np

ROOT = Path.cwd()
ZIP_PATH = ROOT / "Dunnhumby_Dataset" / "dunnhumby_Breakfast-at-the-Frat.zip"
REPORT_DIR = ROOT / "Dataset_Reports"
REPORT_DIR.mkdir(exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP bulunamadı: {ZIP_PATH}")

def load_sheet(archive, excel_name, sheet_name):
    with archive.open(excel_name) as f:
        # İlk satır çalışma kitabının başlığı;
        # ikinci satır gerçek sütun isimleri.
        return pd.read_excel(
            f,
            sheet_name=sheet_name,
            header=1,
            engine="openpyxl"
        )

with zipfile.ZipFile(ZIP_PATH) as archive:
    excel_name = next(
        n for n in archive.namelist()
        if n.lower().endswith(".xlsx")
    )

    print("Reading Dunnhumby Excel...", flush=True)

    transactions = load_sheet(
        archive, excel_name, "dh Transaction Data"
    )
    stores = load_sheet(
        archive, excel_name, "dh Store Lookup"
    )
    products = load_sheet(
        archive, excel_name, "dh Products Lookup"
    )

print("\n=== DATASET DIMENSIONS ===")
for name, df in [
    ("Transactions", transactions),
    ("Stores", stores),
    ("Products", products)
]:
    print(f"{name}: {len(df):,} rows, {len(df.columns)} columns")
    print("Columns:", list(df.columns))

print("\n=== TRANSACTION SAMPLE ===")
print(transactions.head(5).to_string(index=False))

print("\n=== MISSING VALUES ===")
missing = pd.DataFrame({
    "column": transactions.columns,
    "missing": transactions.isna().sum().values,
    "missing_pct": (
        transactions.isna().mean().values * 100
    ).round(4)
})
print(missing.to_string(index=False))

required = [
    "WEEK_END_DATE", "STORE_NUM", "UPC",
    "UNITS", "VISITS", "HHS", "SPEND",
    "PRICE", "BASE_PRICE",
    "FEATURE", "DISPLAY", "TPR_ONLY"
]

absent = [c for c in required if c not in transactions.columns]
if absent:
    raise ValueError(f"Missing expected columns: {absent}")

print("\n=== KEY VALIDATION ===")
key = ["WEEK_END_DATE", "STORE_NUM", "UPC"]

print("Duplicate transaction keys:",
      transactions.duplicated(subset=key).sum())
print("Distinct stores:", transactions["STORE_NUM"].nunique())
print("Distinct products:", transactions["UPC"].nunique())
print("Distinct weeks:", transactions["WEEK_END_DATE"].nunique())
print("Date range:",
      transactions["WEEK_END_DATE"].min(),
      "to",
      transactions["WEEK_END_DATE"].max())

print("\n=== FINANCIAL AND PROMOTION ANALYSIS ===")

numeric_columns = [
    "UNITS", "VISITS", "HHS", "SPEND",
    "PRICE", "BASE_PRICE",
    "FEATURE", "DISPLAY", "TPR_ONLY"
]

for col in numeric_columns:
    transactions[col] = pd.to_numeric(
        transactions[col], errors="coerce"
    )

price = transactions["PRICE"]
base = transactions["BASE_PRICE"]

valid_price = (
    price.notna() & base.notna()
    & (price > 0) & (base > 0)
)

discount = pd.Series(np.nan, index=transactions.index)
discount.loc[valid_price] = (
    (base.loc[valid_price] - price.loc[valid_price])
    / base.loc[valid_price]
)

transactions["DISCOUNT_RATE"] = discount

print("Valid price pairs:", int(valid_price.sum()))
print("PRICE <= 0:", int((price <= 0).sum()))
print("BASE_PRICE <= 0:", int((base <= 0).sum()))
print("PRICE < BASE_PRICE:", int((valid_price & (price < base)).sum()))
print("PRICE = BASE_PRICE:", int((valid_price & (price == base)).sum()))
print("PRICE > BASE_PRICE:", int((valid_price & (price > base)).sum()))

print("\nDiscount distribution (%):")
print(
    (transactions["DISCOUNT_RATE"] * 100)
    .describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    .to_string()
)

print("\nPromotion flag distributions:")
for col in ["FEATURE", "DISPLAY", "TPR_ONLY"]:
    print(f"\n{col}")
    print(transactions[col].value_counts(dropna=False).to_string())

print("\n=== FINANCIAL CONSISTENCY ===")
expected_spend = transactions["UNITS"] * transactions["PRICE"]
valid_spend = (
    expected_spend.notna() &
    transactions["SPEND"].notna()
)
difference = (
    transactions.loc[valid_spend, "SPEND"]
    - expected_spend.loc[valid_spend]
).abs()

print("Spend vs UNITS * PRICE:")
print("Compared rows:", int(valid_spend.sum()))
print("Difference > 0.01:", int((difference > 0.01).sum()))
print("Maximum difference:", difference.max())

print("\n=== LOOKUP KEY COVERAGE ===")
for name, lookup, transaction_key, lookup_key in [
    ("Stores", stores, "STORE_NUM", "STORE_ID"),
    ("Products", products, "UPC", "UPC")
]:
    if lookup_key not in lookup.columns:
        print(f"{name}: missing lookup column {lookup_key}")
        continue

    left = transactions[transaction_key].astype("string").str.strip()
    right = lookup[lookup_key].astype("string").str.strip()

    print(f"{name} lookup duplicate keys:",
          int(right.duplicated().sum()))
    print(f"{name} matched transaction rows:",
          int(left.isin(set(right.dropna())).sum()))
    print(f"{name} unmatched transaction rows:",
          int((~left.isin(set(right.dropna()))).sum()))

print("\n=== SAVE REPORTS ===")
missing.to_csv(
    REPORT_DIR / "dunnhumby_missing_values.csv",
    index=False
)

transactions["DISCOUNT_RATE"].describe(
    percentiles=[.01, .05, .25, .5, .75, .95, .99]
).to_csv(
    REPORT_DIR / "dunnhumby_discount_distribution.csv"
)

print("Reports saved to Dataset_Reports")
print("\nDUNNHUMBY FULL AUDIT COMPLETED")
