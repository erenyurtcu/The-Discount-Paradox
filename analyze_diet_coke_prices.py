from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")

con = duckdb.connect()

print("1/3 - Checking price, quantity and zero sales...", flush=True)

quality = con.execute("""
SELECT
    COUNT(*) AS rows,
    COUNT(*) FILTER (WHERE price IS NULL OR price <= 0) AS invalid_price,
    COUNT(*) FILTER (WHERE qty IS NULL OR qty <= 0) AS invalid_qty,
    COUNT(*) FILTER (WHERE move IS NULL OR move < 0) AS invalid_move,
    COUNT(*) FILTER (WHERE move = 0 AND price > 0 AND qty > 0)
        AS zero_sales_with_valid_price,
    COUNT(DISTINCT qty) AS distinct_package_qty,
    MIN(qty) AS min_qty,
    MAX(qty) AS max_qty
FROM read_parquet(?)
""", [str(src)]).fetchdf()

print(quality.to_string(index=False))

print("2/3 - Comparing consecutive weeks...", flush=True)

con.execute("""
CREATE TEMP TABLE weekly AS
WITH base AS (
    SELECT
        store, week, move, sale,
        price / NULLIF(qty, 0) AS unit_price
    FROM read_parquet(?)
),
lagged AS (
    SELECT *,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_week,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_price,
        LAG(move) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_move
    FROM base
)
SELECT *,
    100.0 * (unit_price / prev_price - 1) AS price_change_pct,
    move - prev_move AS sales_change
FROM lagged
WHERE prev_week = week - 1
  AND unit_price > 0
  AND prev_price > 0
  AND move >= 0
  AND prev_move >= 0
""", [str(src)])

summary = con.execute("""
SELECT
    CASE
        WHEN price_change_pct <= -5 THEN 'Price cut >= 5%'
        WHEN price_change_pct >= 5 THEN 'Price increase >= 5%'
        ELSE 'Price roughly stable'
    END AS price_group,
    COUNT(*) AS observations,
    ROUND(AVG(price_change_pct), 2) AS avg_price_change_pct,
    ROUND(AVG(sales_change), 2) AS avg_unit_sales_change,
    ROUND(AVG(move), 2) AS avg_current_units,
    ROUND(AVG(prev_move), 2) AS avg_previous_units
FROM weekly
GROUP BY 1
ORDER BY 1
""").fetchdf()

print("\n=== OBSERVED WEEK-TO-WEEK COMPARISON ===")
print(summary.to_string(index=False))

print("3/3 - Saving analysis...", flush=True)

summary.to_csv(out / "diet_coke_price_sales_summary.csv", index=False)

con.execute("""
COPY (
    SELECT
        store, week, move, prev_move,
        unit_price, prev_price,
        price_change_pct, sales_change
    FROM weekly
)
TO 'Dominicks_Dataset/Analysis/diet_coke_weekly_changes.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nDIET COKE PRICE-SALES AUDIT COMPLETED")
