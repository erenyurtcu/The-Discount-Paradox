from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_promotion_diagnostic.csv")

con = duckdb.connect()

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        move,
        price / qty AS unit_price,
        COALESCE(NULLIF(TRIM(sale), ''), 'NONE') AS sale_code
    FROM read_parquet(?)
    WHERE price > 0
      AND qty > 0
      AND move > 0
),
lagged AS (
    SELECT
        *,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS previous_week,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS previous_price
    FROM base
),
classified AS (
    SELECT
        *,
        CASE
            WHEN previous_week != week - 1
              OR previous_price IS NULL THEN 'NO_PAIR'
            WHEN unit_price / previous_price <= 0.95 THEN 'PRICE_CUT'
            WHEN unit_price / previous_price >= 1.05 THEN 'PRICE_INCREASE'
            ELSE 'STABLE'
        END AS price_event
    FROM lagged
)
SELECT
    sale_code,
    price_event,
    COUNT(*) AS observations,
    ROUND(AVG(unit_price), 4) AS avg_unit_price,
    ROUND(AVG(move), 2) AS avg_units
FROM classified
GROUP BY sale_code, price_event
ORDER BY sale_code, price_event
""", [str(src)]).fetchdf()

con.close()
df.to_csv(out, index=False)

print("=== PROMOTION DIAGNOSTIC ===")
print(df.to_string(index=False))

print("\n=== TOTAL BY SALE CODE ===")
print(
    df.groupby("sale_code")["observations"]
      .sum()
      .sort_values(ascending=False)
      .to_string()
)

print("\nSaved:", out)
print("PROMOTION DIAGNOSTIC COMPLETED")
