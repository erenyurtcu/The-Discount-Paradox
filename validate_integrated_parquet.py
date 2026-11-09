from pathlib import Path
import duckdb

folder = Path("Dominicks_Dataset/Parquet_Integrated")
files = sorted(folder.glob("*.parquet"))

if len(files) != 28:
    raise RuntimeError(f"Expected 28 files, found {len(files)}")

con = duckdb.connect()
con.execute("SET threads=4")

path = (folder / "*.parquet").resolve().as_posix()

result = con.execute("""
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE financial_eligible) AS eligible_rows,
    COUNT(*) FILTER (WHERE product_matched) AS product_matched,
    COUNT(*) FILTER (WHERE customer_key_matched) AS customer_matched,
    COUNT(*) FILTER (WHERE customer_count IS NOT NULL) AS customer_valid,
    COUNT(DISTINCT category) AS categories
FROM read_parquet(?)
""", [path]).df()

print(result.to_string(index=False))

row = result.iloc[0]
assert int(row["total_rows"]) == 175521485
assert int(row["eligible_rows"]) == 96717530
assert int(row["categories"]) == 28

print("GLOBAL PARQUET VALIDATION PASSED")
con.close()
