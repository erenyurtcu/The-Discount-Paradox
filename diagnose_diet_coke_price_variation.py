from pathlib import Path
import duckdb
import pandas as pd

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/2 - Analyzing weekly store-level price movements...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        qty,
        CASE WHEN price > 0 AND qty > 0
             THEN price / qty ELSE NULL END AS unit_price
    FROM read_parquet(?)
),
lagged AS (
    SELECT
        *,
        LAG(week) OVER w AS prev_week,
        LAG(qty) OVER w AS prev_qty,
        LAG(unit_price) OVER w AS prev_price
    FROM base
    WINDOW w AS (PARTITION BY store ORDER BY week)
),
changes AS (
    SELECT
        store,
        week,
        unit_price / prev_price - 1 AS price_change
    FROM lagged
    WHERE prev_week = week - 1
      AND qty = prev_qty
      AND unit_price > 0
      AND prev_price > 0
),
weekly AS (
    SELECT
        week,
        COUNT(*) AS stores,
        COUNT(*) FILTER (
            WHERE price_change <= -0.05
        ) AS cuts,
        COUNT(*) FILTER (
            WHERE price_change >= 0.05
        ) AS increases,
        COUNT(*) FILTER (
            WHERE ABS(price_change) < 0.05
        ) AS stable
    FROM changes
    GROUP BY week
)
SELECT *
FROM weekly
ORDER BY week
""", [str(src)]).fetchdf()

df["mixed_cut_control"] = (
    (df["cuts"] >= 5) &
    (df["stable"] >= 5)
)

df["mixed_increase_control"] = (
    (df["increases"] >= 5) &
    (df["stable"] >= 5)
)

print("\n=== PRICE MOVEMENT SUMMARY ===")
print("Weeks:", len(df))
print("Average stores per week:", round(df["stores"].mean(), 2))
print("Weeks with >=5 cuts and >=5 stable stores:",
      int(df["mixed_cut_control"].sum()))
print("Weeks with >=5 increases and >=5 stable stores:",
      int(df["mixed_increase_control"].sum()))
print("Weeks with both cuts and increases:",
      int(((df["cuts"] > 0) & (df["increases"] > 0)).sum()))

print("\n=== SAMPLE WEEKS ===")
print(
    df.sort_values(
        ["cuts", "stable"],
        ascending=False
    ).head(10).to_string(index=False)
)

df.to_csv(
    out / "diet_coke_weekly_price_variation.csv",
    index=False
)

print("\n2/2 - Results saved")
print("DIET COKE PRICE VARIATION DIAGNOSTIC COMPLETED")
