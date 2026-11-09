from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Building clean pre-treatment windows...", flush=True)

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
        LAG(week, 3) OVER w AS w3,
        LAG(move, 1) OVER w AS q1,
        LAG(move, 2) OVER w AS q2,
        LAG(move, 3) OVER w AS q3,
        LAG(unit_price, 1) OVER w AS p1,
        LAG(unit_price, 2) OVER w AS p2,
        LAG(unit_price, 3) OVER w AS p3,
        LAG(qty, 1) OVER w AS qty1,
        LAG(qty, 2) OVER w AS qty2,
        LAG(qty, 3) OVER w AS qty3
    FROM base
    WINDOW w AS (PARTITION BY store ORDER BY week)
)
SELECT
    store, week, move, q1, q2, q3,
    unit_price, p1, p2, p3,
    unit_price / p1 - 1 AS event_price_change,
    p1 / p2 - 1 AS pre_price_change_1,
    p2 / p3 - 1 AS pre_price_change_2
FROM lags
WHERE w1 = week - 1
  AND w2 = week - 2
  AND w3 = week - 3
  AND unit_price > 0
  AND p1 > 0 AND p2 > 0 AND p3 > 0
  AND qty = qty1
  AND qty1 = qty2
  AND qty2 = qty3
  AND move >= 0
  AND q1 >= 0 AND q2 >= 0 AND q3 >= 0
""", [str(src)]).fetchdf()

df = df[
    (df["pre_price_change_1"].abs() < 0.02) &
    (df["pre_price_change_2"].abs() < 0.02)
].copy()

df["group"] = np.select(
    [
        df["event_price_change"] <= -0.05,
        df["event_price_change"].abs() < 0.02
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

eligible = counts.index[
    (counts.get("CUT", 0) >= 5) &
    (counts.get("STABLE", 0) >= 5)
]

sample = df[
    df["week"].isin(eligible) &
    df["group"].isin(["CUT", "STABLE"])
].copy()

print("Eligible weeks:", len(eligible))
print("Cut observations:", int((sample["group"] == "CUT").sum()))
print("Stable observations:", int((sample["group"] == "STABLE").sum()))

print("\n2/3 - Checking two pre-event sales movements...", flush=True)

sample["pretrend_early"] = np.log1p(sample["q2"]) - np.log1p(sample["q3"])
sample["pretrend_late"] = np.log1p(sample["q1"]) - np.log1p(sample["q2"])

for variable in ["pretrend_early", "pretrend_late"]:
    weekly = (
        sample.groupby(["week", "group"])[variable]
        .mean()
        .unstack()
        .dropna()
    )

    gap = weekly["CUT"] - weekly["STABLE"]

    print(f"\n{variable}")
    print("Comparable weeks:", len(weekly))
    print("Mean CUT:", round(weekly["CUT"].mean(), 5))
    print("Mean STABLE:", round(weekly["STABLE"].mean(), 5))
    print("Mean gap:", round(gap.mean(), 5))
    print("Median absolute gap:", round(gap.abs().median(), 5))

sample.to_parquet(
    out / "diet_coke_clean_event_candidates.parquet",
    index=False
)

print("\n3/3 - Saved clean event candidates")
print("DIET COKE CLEAN PRETREND DIAGNOSTIC COMPLETED")
