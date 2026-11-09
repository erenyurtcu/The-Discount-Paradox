from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Preparing first-difference panel...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store, week, move, price, qty,
        CASE
            WHEN price > 0 AND qty > 0
            THEN price / qty
            ELSE NULL
        END AS unit_price
    FROM read_parquet(?)
),
lagged AS (
    SELECT
        *,
        LAG(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_week,
        LAG(move) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_move,
        LAG(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_price,
        LAG(qty) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_qty
    FROM base
)
SELECT *
FROM lagged
WHERE prev_week = week - 1
  AND unit_price > 0
  AND prev_price > 0
  AND move >= 0
  AND prev_move >= 0
""", [str(src)]).fetchdf()

df["d_log_sales"] = np.log1p(df["move"]) - np.log1p(df["prev_move"])
df["d_log_price"] = np.log(df["unit_price"]) - np.log(df["prev_price"])

print("Consecutive-week pairs:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

specs = [
    ("FD_all", df),
    ("FD_qty_unchanged", df.loc[df["qty"] == df["prev_qty"]].copy())
]

results = []

print("2/3 - Estimating first-difference models...", flush=True)

for name, sample in specs:
    print(f"Fitting {name}: {len(sample):,} pairs", flush=True)

    model = smf.ols(
        "d_log_sales ~ d_log_price + C(week)",
        data=sample
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["store"]}
    )

    ci = model.conf_int().loc["d_log_price"]

    results.append({
        "model": name,
        "observations": len(sample),
        "coefficient": model.params["d_log_price"],
        "clustered_se": model.bse["d_log_price"],
        "ci_lower": ci.iloc[0],
        "ci_upper": ci.iloc[1],
        "p_value": model.pvalues["d_log_price"],
        "r_squared": model.rsquared
    })

result = pd.DataFrame(results)

print("\n=== FIRST DIFFERENCE RESULTS ===")
print(result.round(5).to_string(index=False))

result.to_csv(
    out / "diet_coke_first_difference_results.csv",
    index=False
)

print("\n3/3 - Results saved")
print("DIET COKE FIRST DIFFERENCE COMPLETED")
