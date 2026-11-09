from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
con = duckdb.connect()

print("1/3 - Checking margin and profit availability...", flush=True)

result = con.execute("""
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE price > 0 AND move > 0) AS valid_sales,
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0 AND margin_pct IS NOT NULL
    ) AS margin_available,
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0 AND margin_pct BETWEEN 0 AND 100
    ) AS margin_0_to_100,
    COUNT(*) FILTER (WHERE margin_pct < 0) AS negative_margin,
    COUNT(*) FILTER (WHERE margin_pct > 100) AS margin_over_100,
    COUNT(*) FILTER (
        WHERE gross_profit IS NOT NULL
    ) AS profit_available
FROM read_parquet(?)
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n2/3 - Margin distribution...", flush=True)

result = con.execute("""
SELECT
    ROUND(MIN(margin_pct), 4) AS min_margin,
    ROUND(QUANTILE_CONT(margin_pct, 0.05), 4) AS p05,
    ROUND(MEDIAN(margin_pct), 4) AS median_margin,
    ROUND(AVG(margin_pct), 4) AS avg_margin,
    ROUND(QUANTILE_CONT(margin_pct, 0.95), 4) AS p95,
    ROUND(MAX(margin_pct), 4) AS max_margin
FROM read_parquet(?)
WHERE price > 0
  AND move > 0
  AND margin_pct IS NOT NULL
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n3/3 - Checking implied unit costs...", flush=True)

result = con.execute("""
WITH financial AS (
    SELECT
        price / qty AS unit_price,
        margin_pct,
        (price / qty) * (1 - margin_pct / 100.0) AS implied_unit_cost
    FROM read_parquet(?)
    WHERE price > 0
      AND qty > 0
      AND move > 0
      AND margin_pct BETWEEN 0 AND 100
)
SELECT
    COUNT(*) AS observations,
    ROUND(AVG(unit_price), 4) AS avg_unit_price,
    ROUND(AVG(implied_unit_cost), 4) AS avg_implied_cost,
    ROUND(MEDIAN(implied_unit_cost), 4) AS median_implied_cost,
    ROUND(MIN(implied_unit_cost), 4) AS min_implied_cost,
    ROUND(MAX(implied_unit_cost), 4) AS max_implied_cost
FROM financial
""", [str(src)]).fetchdf()

print(result.to_string(index=False))
print("\nDIET COKE FINANCIAL QUALITY AUDIT COMPLETED")
