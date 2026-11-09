
from pathlib import Path
import duckdb

file = (
    Path("Dominicks_Dataset")
    / "Movement_Extracted"
    / "wsha.csv"
).resolve().as_posix()

con = duckdb.connect()

print("=== EKSIK UPC: 370071913 ===")

query = """
WITH records AS (
    SELECT
        TRY_CAST(STORE AS INTEGER) AS store,
        TRY_CAST(WEEK AS INTEGER) AS week,
        TRY_CAST(MOVE AS DOUBLE) AS move,
        TRY_CAST(PRICE AS DOUBLE) AS price,
        TRY_CAST(PROFIT AS DOUBLE) AS profit,
        TRIM(CAST(SALE AS VARCHAR)) AS sale,
        TRIM(CAST(OK AS VARCHAR)) AS ok
    FROM read_csv(
        ?,
        header=true,
        all_varchar=true,
        encoding='latin-1'
    )
    WHERE TRIM(UPC) = '370071913'
)
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT store) AS stores,
    MIN(week) AS first_week,
    MAX(week) AS last_week,
    COUNT(*) FILTER (WHERE move > 0) AS positive_move_rows,
    COUNT(*) FILTER (WHERE price > 0) AS positive_price_rows,
    MAX(move) AS max_move,
    MAX(price) AS max_price,
    COUNT(DISTINCT sale) AS distinct_sale_values,
    COUNT(DISTINCT ok) AS distinct_ok_values
FROM records
"""

result = con.execute(query, [file]).df()
print(result.to_string(index=False))

print("\n=== MAGAZA DAGILIMI ===")

stores = con.execute("""
    SELECT
        STORE,
        COUNT(*) AS rows,
        MIN(TRY_CAST(WEEK AS INTEGER)) AS min_week,
        MAX(TRY_CAST(WEEK AS INTEGER)) AS max_week,
        SUM(
            CASE WHEN TRY_CAST(MOVE AS DOUBLE) > 0
                 THEN 1 ELSE 0 END
        ) AS positive_move_rows
    FROM read_csv(
        ?,
        header=true,
        all_varchar=true,
        encoding='latin-1'
    )
    WHERE TRIM(UPC) = '370071913'
    GROUP BY STORE
    ORDER BY rows DESC
""", [file]).df()

print(stores.to_string(index=False))

con.close()
