from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
con = duckdb.connect()

print("1/2 - Inspecting zero-price weeks...", flush=True)

summary = con.execute("""
WITH neighbors AS (
    SELECT
        store, week, move, price, qty, sale,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_week,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_price,
        LEAD(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS next_week,
        LEAD(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS next_price
    FROM read_parquet(?)
)
SELECT
    COUNT(*) AS zero_price_rows,
    COUNT(*) FILTER (
        WHERE prev_week = week - 1 AND prev_price > 0
    ) AS valid_previous_week,
    COUNT(*) FILTER (
        WHERE next_week = week + 1 AND next_price > 0
    ) AS valid_next_week,
    COUNT(*) FILTER (
        WHERE prev_week = week - 1
          AND next_week = week + 1
          AND prev_price > 0
          AND next_price > 0
    ) AS valid_both_neighbors,
    COUNT(*) FILTER (
        WHERE move > 0
    ) AS positive_sales_zero_price
FROM neighbors
WHERE price = 0
""", [str(src)]).fetchdf()

print(summary.to_string(index=False))

print("\n2/2 - Inspecting exceptional records...", flush=True)

exceptions = con.execute("""
SELECT
    store, week, move, price, qty,
    sale, revenue, gross_profit
FROM read_parquet(?)
WHERE price = 0 AND move > 0
ORDER BY store, week
""", [str(src)]).fetchdf()

print(exceptions.to_string(index=False))

print("\nDIET COKE ZERO PRICE DIAGNOSTICS COMPLETED")
