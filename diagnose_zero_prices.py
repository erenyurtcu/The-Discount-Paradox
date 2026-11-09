
from pathlib import Path
import duckdb

file = (
    Path("Dominicks_Dataset")
    / "Movement_Extracted"
    / "wana.csv"
).resolve().as_posix()

con = duckdb.connect()
con.execute("SET threads = 4")
con.execute("SET memory_limit = '4GB'")

print("=== ZERO PRICE DIAGNOSTICS ===", flush=True)

result = con.execute("""
WITH m AS (
    SELECT
        TRY_CAST(MOVE AS DOUBLE) AS move,
        TRY_CAST(PRICE AS DOUBLE) AS price,
        TRY_CAST(OK AS INTEGER) AS ok,
        TRY_CAST(PROFIT AS DOUBLE) AS profit,
        NULLIF(UPPER(TRIM(SALE)), '') AS sale
    FROM read_csv(
        ?,
        header=true,
        all_varchar=true,
        encoding='latin-1',
        strict_mode=true
    )
)
SELECT
    COUNT(*) AS total_rows,

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
        WHERE price IS NULL
    ) AS missing_price,

    COUNT(*) FILTER (
        WHERE price < 0
    ) AS negative_price,

    COUNT(*) FILTER (
        WHERE price = 0 AND ok = 1
    ) AS zero_price_valid_ok,

    COUNT(*) FILTER (
        WHERE price = 0 AND ok = 0
    ) AS zero_price_invalid_ok,

    COUNT(*) FILTER (
        WHERE price = 0 AND sale IS NOT NULL
    ) AS zero_price_promotion,

    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0 AND profit < 0
    ) AS positive_sales_negative_margin

FROM m
""", [file]).df()

print(result.to_string(index=False))

print("\n=== PRICE / MOVE DISTRIBUTION ===")

distribution = con.execute("""
WITH m AS (
    SELECT
        TRY_CAST(MOVE AS DOUBLE) AS move,
        TRY_CAST(PRICE AS DOUBLE) AS price,
        TRY_CAST(OK AS INTEGER) AS ok
    FROM read_csv(
        ?,
        header=true,
        all_varchar=true,
        encoding='latin-1',
        strict_mode=true
    )
)
SELECT
    CASE
        WHEN price IS NULL THEN 'MISSING PRICE'
        WHEN price < 0 THEN 'NEGATIVE PRICE'
        WHEN price = 0 THEN 'ZERO PRICE'
        ELSE 'POSITIVE PRICE'
    END AS price_group,

    CASE
        WHEN move IS NULL THEN 'MISSING MOVE'
        WHEN move = 0 THEN 'ZERO MOVE'
        WHEN move > 0 THEN 'POSITIVE MOVE'
        ELSE 'NEGATIVE MOVE'
    END AS move_group,

    COUNT(*) AS rows,
    COUNT(*) FILTER (WHERE ok = 1) AS valid_ok,
    COUNT(*) FILTER (WHERE ok = 0) AS invalid_ok

FROM m
GROUP BY 1, 2
ORDER BY rows DESC
""", [file]).df()

print(distribution.to_string(index=False))

con.close()
print("\nDIAGNOSTICS COMPLETED")
