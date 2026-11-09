from pathlib import Path
import pandas as pd
import zipfile
import io

root = Path("Dunnhumby_Dataset")
output_dir = root / "Processed"
output_dir.mkdir(parents=True, exist_ok=True)

archives = list(root.rglob("*.zip"))
if not archives:
    raise FileNotFoundError("Dunnhumby ZIP not found")

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
            break

if workbook is None:
    raise FileNotFoundError("Excel workbook not found in ZIP")

stores = pd.read_excel(
    workbook,
    sheet_name="dh Store Lookup",
    header=1
)

stores.columns = stores.columns.str.strip()

# Check whether duplicate stores differ in any field
# other than their segment classification.
non_segment = [
    c for c in stores.columns
    if c != "SEG_VALUE_NAME"
]

for store_id, group in stores.groupby("STORE_ID"):
    if len(group) > 1:
        for col in non_segment:
            if group[col].nunique(dropna=False) > 1:
                raise RuntimeError(
                    f"Conflicting store attributes: "
                    f"{store_id}, {col}"
                )

def merge_segments(series):
    values = sorted(
        set(series.dropna().astype(str).str.strip())
    )
    return "|".join(values)

segments = (
    stores.groupby("STORE_ID")["SEG_VALUE_NAME"]
    .apply(merge_segments)
    .reset_index(name="SEGMENT_ORIGINAL")
)

clean = (
    stores.drop(columns=["SEG_VALUE_NAME"])
    .drop_duplicates(subset=["STORE_ID"])
    .merge(segments, on="STORE_ID", how="left")
)

clean["SEG_VALUE_NAME"] = clean["SEGMENT_ORIGINAL"].apply(
    lambda x: "MULTIPLE" if "|" in x else x
)

assert len(clean) == 77
assert clean["STORE_ID"].is_unique

output = output_dir / "stores_clean.parquet"
clean.to_parquet(
    output, index=False, compression="zstd"
)

print("=== DUNNHUMBY CLEAN STORES ===")
print("Original rows:", len(stores))
print("Unique stores:", len(clean))
print("Multiple-segment stores:")
print(
    clean.loc[
        clean["SEG_VALUE_NAME"] == "MULTIPLE",
        ["STORE_ID", "STORE_NAME", "SEGMENT_ORIGINAL"]
    ].to_string(index=False)
)
print("Saved:", output)
print("DUNNHUMBY STORE CLEANING PASSED")
