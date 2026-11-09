from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
con = duckdb.connect()

print("1/3 - Cost distribution including negative margins...", flush=True)

result = con.execute("""
WITH financial AS (
    SELECT
        store, week, move, price, qty, margin_pct,
        price / qty AS unit_price,
        (price / qty) * (1 - margin_pct / 100.0) AS implied_cost
    FROM read_parquet(?)
    WHERE price > 0
      AND qty > 0
      AND move > 0
      AND margin_pct IS NOT NULL
      AND margin_pct <= 100
)
SELECT
    COUNT(*) AS observations,
    COUNT(*) FILTER (WHERE margin_pct < 0) AS negative_margin_rows,
    ROUND(AVG(implied_cost), 4) AS avg_cost,
    ROUND(MEDIAN(implied_cost), 4) AS median_cost,
    ROUND(QUANTILE_CONT(implied_cost, 0.05), 4) AS p05_cost,
    ROUND(QUANTILE_CONT(implied_cost, 0.95), 4) AS p95_cost,
    ROUND(MIN(implied_cost), 4) AS min_cost,
    ROUND(MAX(implied_cost), 4) AS max_cost
FROM financial
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n2/3 - Checking missing gross profit...", flush=True)

result = con.execute("""
SELECT
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0 AND gross_profit IS NULL
    ) AS missing_profit,
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0
          AND gross_profit IS NULL AND margin_pct < 0
    ) AS missing_profit_negative_margin,
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0
          AND gross_profit IS NOT NULL AND margin_pct < 0
    ) AS available_profit_negative_margin,
    COUNT(*) FILTER (
        WHERE price > 0 AND move > 0
          AND gross_profit IS NULL AND margin_pct >= 0
    ) AS missing_profit_nonnegative_margin
FROM read_parquet(?)
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\n3/3 - Validating profit calculations...", flush=True)

result = con.execute("""
WITH checked AS (
    SELECT
        gross_profit,
        (price * move / qty) * margin_pct / 100.0
            AS expected_profit
    FROM read_parquet(?)
    WHERE price > 0
      AND qty > 0
      AND move > 0
      AND margin_pct IS NOT NULL
      AND gross_profit IS NOT NULL
)
SELECT
    COUNT(*) AS checked_rows,
    ROUND(AVG(ABS(gross_profit - expected_profit)), 6)
        AS mean_absolute_error,
    ROUND(MAX(ABS(gross_profit - expected_profit)), 6)
        AS max_absolute_error,
    COUNT(*) FILTER (
        WHERE ABS(gross_profit - expected_profit) > 0.01
    ) AS mismatches_over_one_cent
FROM checked
""", [str(src)]).fetchdf()

print(result.to_string(index=False))

print("\nDIET COKE COST VALIDATION COMPLETED")
