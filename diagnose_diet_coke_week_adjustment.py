from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_week_adjusted_results.csv")

con = duckdb.connect()

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        LN(price / qty) AS log_price,
        LN(move) AS log_sales
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
        ) AS prev_week,
        LAG(log_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_log_price,
        LAG(log_sales) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_log_sales
    FROM base
)
SELECT
    store,
    week,
    log_price - prev_log_price AS dlog_price,
    log_sales - prev_log_sales AS dlog_sales
FROM lagged
WHERE prev_week = week - 1
""", [str(src)]).fetchdf()

con.close()

print("1/3 - Preparing weekly adjusted changes...", flush=True)
print("Pairs:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

df["adjusted_price"] = (
    df["dlog_price"]
    - df.groupby("week")["dlog_price"].transform("mean")
)

df["adjusted_sales"] = (
    df["dlog_sales"]
    - df.groupby("week")["dlog_sales"].transform("mean")
)

print("\n2/3 - Estimating associations...", flush=True)

results = []

for name, xcol, ycol in [
    ("Raw first differences", "dlog_price", "dlog_sales"),
    ("Week-adjusted differences", "adjusted_price", "adjusted_sales")
]:
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    model = sm.OLS(
        y,
        sm.add_constant(x)
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store"]}
    )

    ci = model.conf_int()[1]

    results.append({
        "model": name,
        "observations": len(df),
        "coefficient": model.params[1],
        "clustered_se": model.bse[1],
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "p_value": model.pvalues[1],
        "r_squared": model.rsquared,
        "price_std": np.std(x)
    })

result = pd.DataFrame(results)

print("\n=== WEEKLY ADJUSTMENT COMPARISON ===")
print(result.round(5).to_string(index=False))

result.to_csv(out, index=False)

print("\n3/3 - Results saved:", out)
print("WEEKLY ADJUSTMENT DIAGNOSTIC COMPLETED")
