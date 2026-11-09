from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Building candidate treatment-control weeks...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store, week, move, qty,
        CASE WHEN price > 0 AND qty > 0
             THEN price / qty ELSE NULL END AS unit_price
    FROM read_parquet(?)
),
lags AS (
    SELECT *,
        LAG(week, 1) OVER w AS w1,
        LAG(week, 2) OVER w AS w2,
        LAG(move, 1) OVER w AS q1,
        LAG(move, 2) OVER w AS q2,
        LAG(unit_price, 1) OVER w AS p1,
        LAG(unit_price, 2) OVER w AS p2,
        LAG(qty, 1) OVER w AS qty1,
        LAG(qty, 2) OVER w AS qty2
    FROM base
    WINDOW w AS (PARTITION BY store ORDER BY week)
)
SELECT
    store, week, move, q1, q2,
    unit_price, p1, p2,
    unit_price / p1 - 1 AS price_change,
    LOG(1 + q1) - LOG(1 + q2) AS pre_sales_change
FROM lags
WHERE w1 = week - 1
  AND w2 = week - 2
  AND unit_price > 0
  AND p1 > 0
  AND p2 > 0
  AND qty = qty1
  AND qty1 = qty2
  AND move >= 0
  AND q1 >= 0
  AND q2 >= 0
""", [str(src)]).fetchdf()

df["group"] = np.select(
    [
        df["price_change"] <= -0.05,
        df["price_change"].abs() < 0.05
    ],
    ["CUT", "STABLE"],
    default="OTHER"
)

counts = (
    df[df["group"].isin(["CUT", "STABLE"])]
    .groupby(["week", "group"])
    .size()
    .unstack(fill_value=0)
)

eligible_weeks = counts.index[
    (counts.get("CUT", 0) >= 5) &
    (counts.get("STABLE", 0) >= 5)
]

sample = df[
    df["week"].isin(eligible_weeks) &
    df["group"].isin(["CUT", "STABLE"])
].copy()

print("Eligible mixed weeks:", len(eligible_weeks))
print("Cut observations:", (sample["group"] == "CUT").sum())
print("Stable observations:", (sample["group"] == "STABLE").sum())

print("\n2/3 - Comparing pre-treatment sales trends...", flush=True)

weekly = (
    sample.groupby(["week", "group"])["pre_sales_change"]
    .mean()
    .unstack()
    .dropna()
)

weekly["pretrend_gap"] = weekly["CUT"] - weekly["STABLE"]

print("\n=== PRE-TREND DIAGNOSTIC ===")
print("Comparable weeks:", len(weekly))
print("Mean CUT pretrend:", round(weekly["CUT"].mean(), 5))
print("Mean STABLE pretrend:", round(weekly["STABLE"].mean(), 5))
print("Mean pretrend gap:", round(weekly["pretrend_gap"].mean(), 5))
print("Median absolute gap:",
      round(weekly["pretrend_gap"].abs().median(), 5))

print("\n=== SAMPLE WEEKLY COMPARISONS ===")
print(weekly.head(12).round(5).to_string())

weekly.to_csv(out / "diet_coke_pretrend_weekly.csv")
sample.to_parquet(
    out / "diet_coke_candidate_did_sample.parquet",
    index=False
)

print("\n3/3 - Files saved")
print("DIET COKE PRETREND DIAGNOSTIC COMPLETED")
