from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
con = duckdb.connect()

print("1/3 - QTY and price quality", flush=True)

result = con.execute("""
SELECT
    qty,
    COUNT(*) AS rows,
    COUNT(*) FILTER (WHERE price > 0) AS positive_prices,
    COUNT(*) FILTER (WHERE move = 0) AS zero_sales,
    ROUND(AVG(price) FILTER (WHERE price > 0), 4) AS avg_price,
    ROUND(AVG(unit_price) FILTER (WHERE unit_price > 0), 4)
        AS avg_unit_price
FROM read_parquet(?)
GROUP BY qty
ORDER BY qty
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n2/3 - QTY changes between consecutive weeks", flush=True)

result = con.execute("""
WITH lagged AS (
    SELECT
        store, week, qty, price, unit_price, move,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_week,
        LAG(qty) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_qty,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_unit_price
    FROM read_parquet(?)
)
SELECT
    CASE
        WHEN qty = prev_qty THEN 'QTY unchanged'
        ELSE 'QTY changed'
    END AS qty_status,
    COUNT(*) AS pairs,
    COUNT(*) FILTER (
        WHERE unit_price > 0 AND prev_unit_price > 0
    ) AS valid_price_pairs,
    COUNT(*) FILTER (
        WHERE unit_price > 0
          AND prev_unit_price > 0
          AND ABS(unit_price / prev_unit_price - 1) >= 0.05
    ) AS price_changes_5pct
FROM lagged
WHERE prev_week = week - 1
GROUP BY 1
ORDER BY 1
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n3/3 - Missing price patterns", flush=True)

result = con.execute("""
SELECT
    CASE
        WHEN price IS NULL THEN 'NULL price'
        WHEN price = 0 THEN 'Zero price'
        WHEN price < 0 THEN 'Negative price'
        ELSE 'Positive price'
    END AS price_status,
    COUNT(*) AS rows,
    COUNT(*) FILTER (WHERE move = 0) AS zero_sales,
    COUNT(*) FILTER (WHERE move > 0) AS positive_sales,
    COUNT(DISTINCT store) AS stores,
    COUNT(DISTINCT week) AS weeks
FROM read_parquet(?)
GROUP BY 1
ORDER BY 1
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\nDIET COKE PRICE QUALITY DIAGNOSTICS COMPLETED")
