from pathlib import Path
import pandas as pd
import zipfile
import io

root = Path("Dunnhumby_Dataset")
files = list(root.rglob("*.xlsx"))

if not files:
    for archive in root.rglob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            names = [
                n for n in z.namelist()
                if n.lower().endswith(".xlsx")
                and not Path(n).name.startswith("~$")
            ]
            if names:
                with z.open(names[0]) as f:
                    content = io.BytesIO(f.read())
                files = [content]
                print("Excel extracted from ZIP:", names[0])
                break

if not files:
    raise FileNotFoundError("Dunnhumby Excel workbook not found")

source = files[0]
print("Source:", source)

xls = pd.ExcelFile(source)
print("Sheets:", xls.sheet_names)

sheet = next(
    s for s in xls.sheet_names
    if "Store Lookup" in s
)

stores = pd.read_excel(
    xls, sheet_name=sheet, header=1
)

stores.columns = stores.columns.str.strip()

print("\n=== STORE LOOKUP ===")
print("Rows:", len(stores))
print("Columns:", stores.columns.tolist())

if "STORE_ID" not in stores.columns:
    raise RuntimeError("STORE_ID column missing")

duplicates = stores[
    stores.duplicated("STORE_ID", keep=False)
].sort_values("STORE_ID")

print("\n=== DUPLICATE STORE RECORDS ===")
print(duplicates.to_string(index=False))

print("\nDuplicate rows:", len(duplicates))
print("Duplicate store IDs:", duplicates["STORE_ID"].nunique())
print("Unique stores:", stores["STORE_ID"].nunique())

print("\nDUNNHUMBY STORE AUDIT COMPLETED")
