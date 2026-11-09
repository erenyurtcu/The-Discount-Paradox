
from pathlib import Path
import duckdb
import sys
import time

ROOT = Path.cwd()
DATA = ROOT / "Dominicks_Dataset"
MOVEMENT = DATA / "Movement_Extracted"
OUTPUT = DATA / "Parquet_Integrated"
OUTPUT.mkdir(parents=True, exist_ok=True)

category = sys.argv[1].lower() if len(sys.argv) > 1 else "ana"

if not category.isalpha() or len(category) != 3:
    raise ValueError("Category must be a three-letter code.")

movement_file = MOVEMENT / f"w{category}.csv"
upc_file = DATA / f"upc{category}.csv"
customer_file = DATA / "Auxiliary" / "customer_weekly.csv"

for file in [movement_file, upc_file, customer_file]:
    if not file.is_file():
        raise FileNotFoundError(f"Missing file: {file}")

output_file = OUTPUT / f"{category}.parquet"
temporary_file = OUTPUT / f"{category}.parquet.tmp"

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")
con.execute("SET preserve_insertion_order = false")

def path_sql(path):
    # Safely quote a path for DuckDB SQL.
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"

def csv_sql(path):
    return (
        "read_csv("
        + path_sql(path)
        + ", header=true, all_varchar=true,"
        + " encoding='latin-1', strict_mode=true)"
    )

started = time.time()

print(f"Category: {category}", flush=True)
print("Preparing product and customer lookup tables...", flush=True)

# Metadata keys are unique within each category,
# but verify this again before joining.
con.execute(f"""
CREATE TEMP TABLE products AS
SELECT
    TRIM(UPC) AS upc,
    DESCRIP AS product_description,
    SIZE AS product_size,
    COM_CODE AS commodity_code
FROM {csv_sql(upc_file)}
""")

duplicate_products = con.execute("""
SELECT COUNT(*) FROM (
    SELECT upc
    FROM products
    GROUP BY upc
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

if duplicate_products:
    raise RuntimeError(
        f"Duplicate product keys: {duplicate_products}"
    )

# Customer counts were prepared from ccount.dta.
con.execute(f"""
CREATE TEMP TABLE customers AS
SELECT
    TRY_CAST(store AS INTEGER) AS store,
    TRY_CAST(week AS INTEGER) AS week,
    TRY_CAST(customer_count AS DOUBLE) AS customer_count,
    TRY_CAST(observed_days AS INTEGER) AS observed_days,
    TRY_CAST(complete_week AS BOOLEAN) AS complete_week
FROM {csv_sql(customer_file)}
""")

duplicate_customers = con.execute("""
SELECT COUNT(*) FROM (
    SELECT store, week
    FROM customers
    GROUP BY store, week
    HAVING COUNT(*) > 1
)
""").fetchone()[0]

if duplicate_customers:
    raise RuntimeError(
        f"Duplicate customer store-week keys: {duplicate_customers}"
    )

print("Lookup keys verified.", flush=True)
print("Building integrated Parquet...", flush=True)

# Do not create a VIEW with parameter placeholders.
# This SQL uses escaped file paths instead.
query = f"""
COPY (
    WITH movement AS (
        SELECT
            '{category}' AS category,
            TRIM(UPC) AS upc,
            TRY_CAST(STORE AS INTEGER) AS store,
            TRY_CAST(WEEK AS INTEGER) AS week,
            TRY_CAST(MOVE AS DOUBLE) AS move,
            TRY_CAST(QTY AS DOUBLE) AS qty,
            TRY_CAST(PRICE AS DOUBLE) AS price,
            TRY_CAST(PROFIT AS DOUBLE) AS margin_pct,
            TRY_CAST(OK AS INTEGER) AS ok,
            NULLIF(TRIM(SALE), '') AS sale
        FROM {csv_sql(movement_file)}
    )
    SELECT
        m.*,

        p.product_description,
        p.product_size,
        p.commodity_code,
        (p.upc IS NOT NULL) AS product_matched,

        c.customer_count,
        c.observed_days,
        c.complete_week,
        (c.store IS NOT NULL) AS customer_key_matched,

        (
            m.ok = 1
            AND m.price > 0
            AND m.move > 0
            AND m.qty > 0
            AND m.margin_pct IS NOT NULL
        ) AS financial_eligible,

        CASE
            WHEN m.ok = 1
             AND m.price > 0
             AND m.move > 0
             AND m.qty > 0
             AND m.margin_pct IS NOT NULL
            THEN m.price * m.move / m.qty
            ELSE NULL
        END AS revenue,

        CASE
            WHEN m.ok = 1
             AND m.price > 0
             AND m.move > 0
             AND m.qty > 0
             AND m.margin_pct IS NOT NULL
            THEN
                (m.price * m.move / m.qty)
                * (m.margin_pct / 100.0)
            ELSE NULL
        END AS gross_profit

    FROM movement m
    LEFT JOIN products p
        ON m.upc = p.upc
    LEFT JOIN customers c
        ON m.store = c.store
       AND m.week = c.week
)
TO {path_sql(temporary_file)}
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

try:
    if temporary_file.exists():
        temporary_file.unlink()

    con.execute(query)

    print("Parquet created. Validating...", flush=True)

    result = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(*) FILTER (
            WHERE product_matched
        ) AS product_matched,
        COUNT(*) FILTER (
            WHERE customer_key_matched
        ) AS customer_key_matched,
        COUNT(*) FILTER (
            WHERE customer_count IS NOT NULL
        ) AS valid_customer_count,
        COUNT(*) FILTER (
            WHERE financial_eligible
        ) AS financially_usable,
        COUNT(*) FILTER (
            WHERE revenue IS NOT NULL
        ) AS revenue_calculated,
        COUNT(*) FILTER (
            WHERE gross_profit IS NOT NULL
        ) AS profit_calculated
    FROM read_parquet({path_sql(temporary_file)})
    """).df()

    print("\n=== INTEGRATED PARQUET VALIDATION ===")
    print(result.to_string(index=False))

    expected_rows = {
        "ana": 7339217
    }

    actual_rows = int(result.iloc[0]["total_rows"])

    if category in expected_rows:
        if actual_rows != expected_rows[category]:
            raise RuntimeError(
                f"Row mismatch: expected "
                f"{expected_rows[category]}, got {actual_rows}"
            )

    if category == "ana":
        usable = int(result.iloc[0]["financially_usable"])
        if usable != 3040172:
            raise RuntimeError(
                f"Financial eligibility mismatch: {usable}"
            )

    # Only publish after successful validation.
    temporary_file.replace(output_file)

    print(f"\nSUCCESS: {output_file}")
    print(
        f"File size: "
        f"{output_file.stat().st_size / 1024**2:.2f} MB"
    )
    print(
        f"Elapsed: {time.time() - started:.2f} seconds"
    )

finally:
    con.close()
