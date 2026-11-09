from pathlib import Path
import duckdb

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_store_variation.csv")

con = duckdb.connect()

print("Analyzing store-level price variation...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        move,
        price / qty AS unit_price
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
)
SELECT
    store,
    COUNT(*) AS observations,
    COUNT(DISTINCT week) AS weeks,
    ROUND(AVG(unit_price), 4) AS avg_price,
    ROUND(STDDEV_SAMP(unit_price), 4) AS price_std,
    COUNT(*) FILTER (
        WHERE previous_week = week - 1
          AND ABS(unit_price / previous_price - 1) >= 0.05
    ) AS substantial_price_changes,
    ROUND(AVG(move), 2) AS avg_weekly_units
FROM lagged
GROUP BY store
ORDER BY store
""", [str(src)]).fetchdf()

df.to_csv(out, index=False)

print("\n=== STORE VARIATION SUMMARY ===")
print(df[[
    "observations",
    "price_std",
    "substantial_price_changes"
]].describe().round(2).to_string())

print("\nStores with >=100 observations:",
      int((df["observations"] >= 100).sum()))

print("Stores with >=30 substantial price changes:",
      int((df["substantial_price_changes"] >= 30).sum()))

print("\nSaved:", out)
print("STORE VARIATION ANALYSIS COMPLETED")
