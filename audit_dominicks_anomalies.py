
from pathlib import Path
import duckdb
import pandas as pd

root = Path.cwd()
movement_dir = root / "Dominicks_Dataset" / "Movement_Extracted"
report_dir = root / "Dominicks_Dataset" / "Quality_Reports"
report_dir.mkdir(parents=True, exist_ok=True)

files = sorted(movement_dir.glob("w*.csv"))

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

results = []

for index, file in enumerate(files, 1):
    category = file.stem[1:]

    query = """
    WITH m AS (
        SELECT
            TRY_CAST(OK AS INTEGER) AS ok,
            TRY_CAST(PRICE AS DOUBLE) AS price,
            TRY_CAST(MOVE AS DOUBLE) AS move,
            TRY_CAST(QTY AS DOUBLE) AS qty,
            TRY_CAST(PROFIT AS DOUBLE) AS profit
        FROM read_csv(
            ?, header=true, all_varchar=true,
            encoding='latin-1', strict_mode=true
        )
    )
    SELECT
        COUNT(*) FILTER (
            WHERE price = 0 AND move > 0
        ) AS zero_price_positive_move,

        COUNT(*) FILTER (
            WHERE price = 0 AND move > 0 AND ok = 1
        ) AS zero_price_positive_move_ok1,

        COUNT(*) FILTER (
            WHERE price = 0 AND move > 0 AND ok = 0
        ) AS zero_price_positive_move_ok0,

        COUNT(*) FILTER (
            WHERE price = 0 AND move > 0
              AND (ok IS NULL OR ok NOT IN (0,1))
        ) AS zero_price_positive_move_other_ok,

        COUNT(*) FILTER (
            WHERE ok IS NULL OR ok NOT IN (0,1)
        ) AS unexpected_ok,

        COUNT(*) FILTER (
            WHERE ok IS NULL
              AND price IS NULL
              AND move IS NULL
              AND qty IS NULL
              AND profit IS NULL
        ) AS all_financial_fields_missing,

        COUNT(*) FILTER (
            WHERE ok = 1 AND price > 0
              AND move > 0 AND qty > 0
              AND profit < 0
        ) AS valid_negative_margin

    FROM m
    """

    row = con.execute(
        query, [file.resolve().as_posix()]
    ).df().iloc[0].to_dict()

    row["category"] = category
    results.append(row)

    print(f"[{index}/{len(files)}] {category} completed", flush=True)

con.close()

df = pd.DataFrame(results)
columns = [c for c in df.columns if c != "category"]

print("\n=== ANOMALY SUMMARY ===")
print(df[["category"] + columns].to_string(index=False))

print("\n=== TOTALS ===")
for col in columns:
    print(f"{col}: {int(df[col].sum()):,}")

output = report_dir / "dominicks_anomaly_summary.csv"
df.to_csv(output, index=False)

print(f"\nSaved: {output}")
print("ANOMALY AUDIT COMPLETED")
