
from pathlib import Path
import duckdb
import pandas as pd
import time

ROOT = Path.cwd()
MOVEMENT_DIR = ROOT / "Dominicks_Dataset" / "Movement_Extracted"
REPORT_DIR = ROOT / "Dominicks_Dataset" / "Quality_Reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(MOVEMENT_DIR.glob("w*.csv"))

if len(files) != 28:
    raise RuntimeError(
        f"Expected 28 movement CSV files, found {len(files)}"
    )

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

results = []
started = time.time()

print(f"Auditing {len(files)} categories...", flush=True)

for index, file in enumerate(files, 1):
    category = file.stem.removeprefix("w")
    category_start = time.time()

    print(
        f"[{index}/28] {category}: processing...",
        flush=True
    )

    query = """
    WITH raw AS (
        SELECT
            TRY_CAST(MOVE AS DOUBLE) AS move,
            TRY_CAST(PRICE AS DOUBLE) AS price,
            TRY_CAST(QTY AS DOUBLE) AS qty,
            TRY_CAST(PROFIT AS DOUBLE) AS profit,
            TRY_CAST(OK AS INTEGER) AS ok,
            NULLIF(UPPER(TRIM(SALE)), '') AS sale
        FROM read_csv(
            ?, header=true, all_varchar=true,
            encoding='latin-1', strict_mode=true
        )
    )
    SELECT
        COUNT(*) AS total_rows,

        COUNT(*) FILTER (
            WHERE ok = 1
        ) AS valid_ok,

        COUNT(*) FILTER (
            WHERE ok = 0
        ) AS invalid_ok,

        COUNT(*) FILTER (
            WHERE ok IS NULL OR ok NOT IN (0,1)
        ) AS unexpected_ok,

        COUNT(*) FILTER (
            WHERE qty IS NULL OR qty <= 0
        ) AS invalid_qty,

        COUNT(*) FILTER (
            WHERE price IS NULL
        ) AS missing_price,

        COUNT(*) FILTER (
            WHERE price = 0 AND move = 0
        ) AS zero_price_zero_move,

        COUNT(*) FILTER (
            WHERE price = 0 AND move > 0
        ) AS zero_price_positive_move,

        COUNT(*) FILTER (
            WHERE price > 0 AND move = 0
        ) AS positive_price_zero_move,

        COUNT(*) FILTER (
            WHERE price > 0 AND move > 0
        ) AS positive_price_positive_move,

        COUNT(*) FILTER (
            WHERE price < 0
        ) AS negative_price,

        COUNT(*) FILTER (
            WHERE move IS NULL OR move < 0
        ) AS invalid_move,

        COUNT(*) FILTER (
            WHERE profit IS NULL
        ) AS missing_profit,

        COUNT(*) FILTER (
            WHERE profit < 0
        ) AS negative_margin,

        COUNT(*) FILTER (
            WHERE profit > 100
        ) AS margin_above_100,

        COUNT(*) FILTER (
            WHERE ok = 1
              AND price > 0
              AND move > 0
              AND qty > 0
              AND profit IS NOT NULL
        ) AS financially_usable_rows,

        COUNT(*) FILTER (
            WHERE sale IS NOT NULL
        ) AS promotion_coded_rows,

        COUNT(DISTINCT qty) AS distinct_qty_values,

        MIN(profit) AS min_profit,
        MAX(profit) AS max_profit

    FROM raw
    """

    row = con.execute(
        query, [file.resolve().as_posix()]
    ).df().iloc[0].to_dict()

    row["category"] = category
    row["file"] = file.name
    row["seconds"] = round(time.time() - category_start, 2)

    results.append(row)

    # Her kategori bittiğinde ara raporu kaydet.
    pd.DataFrame(results).to_csv(
        REPORT_DIR / "financial_audit_all_categories.csv",
        index=False
    )

    print(
        f"    rows={int(row['total_rows']):,} | "
        f"usable={int(row['financially_usable_rows']):,} | "
        f"seconds={row['seconds']}",
        flush=True
    )

con.close()

df = pd.DataFrame(results)

count_columns = [
    "total_rows",
    "valid_ok",
    "invalid_ok",
    "unexpected_ok",
    "invalid_qty",
    "missing_price",
    "zero_price_zero_move",
    "zero_price_positive_move",
    "positive_price_zero_move",
    "positive_price_positive_move",
    "negative_price",
    "invalid_move",
    "missing_profit",
    "negative_margin",
    "margin_above_100",
    "financially_usable_rows",
    "promotion_coded_rows"
]

print("\n=== GLOBAL FINANCIAL SUMMARY ===")

for column in count_columns:
    print(f"{column}: {int(df[column].sum()):,}")

print(
    "\nFinancial usability rate:",
    round(
        df["financially_usable_rows"].sum()
        / df["total_rows"].sum() * 100,
        4
    ),
    "%"
)

print(
    "\nTotal execution time:",
    round(time.time() - started, 2),
    "seconds"
)

print(
    "\nSaved:",
    REPORT_DIR / "financial_audit_all_categories.csv"
)

print("\nDOMINICKS FINANCIAL AUDIT COMPLETED")
