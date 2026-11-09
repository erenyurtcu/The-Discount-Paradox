from pathlib import Path
import pandas as pd
import duckdb
import zipfile
import io
import time

started = time.time()

root = Path("Dunnhumby_Dataset")
output_dir = root / "Processed"
output_dir.mkdir(parents=True, exist_ok=True)

stores_file = output_dir / "stores_clean.parquet"
if not stores_file.exists():
    raise FileNotFoundError(stores_file)

archives = list(root.rglob("*.zip"))
workbook = None

for archive in archives:
    with zipfile.ZipFile(archive) as z:
        matches = [
            n for n in z.namelist()
            if n.lower().endswith(".xlsx")
            and not Path(n).name.startswith("~$")
        ]
        if matches:
            workbook = io.BytesIO(z.read(matches[0]))
            print("Workbook:", matches[0])
            break

if workbook is None:
    raise FileNotFoundError("Dunnhumby workbook not found")

xls = pd.ExcelFile(workbook)

print("Reading transaction and product sheets...", flush=True)

transactions = pd.read_excel(
    xls, sheet_name="dh Transaction Data", header=1
)
products = pd.read_excel(
    xls, sheet_name="dh Products Lookup", header=1
)

transactions.columns = transactions.columns.str.strip()
products.columns = products.columns.str.strip()

expected_transactions = 524950

assert len(transactions) == expected_transactions
assert products["UPC"].is_unique

# Keep UPC as a nullable string key to avoid float joins.
def normalize_upc(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise RuntimeError("Invalid UPC key encountered")
    if (numeric % 1 != 0).any():
        raise RuntimeError("Non-integer UPC key encountered")
    return numeric.astype("int64").astype("string")

transactions["UPC"] = normalize_upc(transactions["UPC"])
products["UPC"] = normalize_upc(products["UPC"])

transactions["STORE_NUM"] = pd.to_numeric(
    transactions["STORE_NUM"], errors="raise"
).astype("int64")

stores = pd.read_parquet(stores_file)
stores["STORE_ID"] = pd.to_numeric(
    stores["STORE_ID"], errors="raise"
).astype("int64")

assert stores["STORE_ID"].is_unique

print("Joining tables...", flush=True)

merged = transactions.merge(
    products,
    on="UPC",
    how="left",
    validate="many_to_one",
    indicator="product_join"
)

merged["product_matched"] = (
    merged["product_join"] == "both"
)
merged = merged.drop(columns=["product_join"])

merged = merged.merge(
    stores,
    left_on="STORE_NUM",
    right_on="STORE_ID",
    how="left",
    validate="many_to_one",
    indicator="store_join"
)

merged["store_matched"] = (
    merged["store_join"] == "both"
)
merged = merged.drop(columns=["store_join"])

assert len(merged) == expected_transactions

for column in [
    "PRICE", "BASE_PRICE", "UNITS", "SPEND",
    "FEATURE", "DISPLAY", "TPR_ONLY"
]:
    merged[column] = pd.to_numeric(
        merged[column], errors="coerce"
    )

merged["price_pair_valid"] = (
    (merged["PRICE"] > 0)
    & (merged["BASE_PRICE"] > 0)
)

merged["discount_rate"] = (
    1 - merged["PRICE"] / merged["BASE_PRICE"]
).where(merged["price_pair_valid"])

merged["is_discounted"] = (
    merged["price_pair_valid"]
    & (merged["PRICE"] < merged["BASE_PRICE"])
)

merged["is_price_premium"] = (
    merged["price_pair_valid"]
    & (merged["PRICE"] > merged["BASE_PRICE"])
)

merged["feature_flag"] = merged["FEATURE"].eq(1)
merged["display_flag"] = merged["DISPLAY"].eq(1)
merged["tpr_only_flag"] = merged["TPR_ONLY"].eq(1)

merged["revenue_check"] = (
    merged["UNITS"] * merged["PRICE"]
)

merged["spend_consistent"] = (
    (merged["SPEND"] - merged["revenue_check"]).abs()
    <= 0.01
).where(
    merged["SPEND"].notna()
    & merged["revenue_check"].notna()
)

# An observed discount is not a randomized intervention.
merged["observed_price_intervention"] = (
    merged["is_discounted"]
)

output = output_dir / "dunnhumby_integrated.parquet"

merged.to_parquet(
    output,
    index=False,
    compression="zstd"
)

print("\n=== DUNNHUMBY INTEGRATION VALIDATION ===")
print("Total rows:", f"{len(merged):,}")
print("Product matched:", f"{merged['product_matched'].sum():,}")
print("Store matched:", f"{merged['store_matched'].sum():,}")
print("Valid price pairs:", f"{merged['price_pair_valid'].sum():,}")
print("Discounted rows:", f"{merged['is_discounted'].sum():,}")
print("Premium price rows:", f"{merged['is_price_premium'].sum():,}")
print("Spend consistent:", f"{merged['spend_consistent'].sum():,}")
print("Missing discount rates:", f"{merged['discount_rate'].isna().sum():,}")
print("File size MB:", round(output.stat().st_size / 1024**2, 2))
print("Elapsed seconds:", round(time.time() - started, 2))

assert len(merged) == 524950
assert merged["product_matched"].all()
assert merged["store_matched"].all()

print("\nDUNNHUMBY INTEGRATION PASSED")
