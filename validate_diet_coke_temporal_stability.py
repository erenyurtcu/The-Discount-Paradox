from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_temporal_stability.csv")

con = duckdb.connect()

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        LN(price / qty) AS log_price,
        LN(move) AS log_sales
    FROM read_parquet(?)
    WHERE price > 0 AND qty > 0 AND move > 0
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

median_week = int(df["week"].median())
print("Temporal split week:", median_week)

def fit_model(part, label):
    part = part.copy()

    part["x"] = (
        part["dlog_price"]
        - part.groupby("week")["dlog_price"].transform("mean")
    )
    part["y"] = (
        part["dlog_sales"]
        - part.groupby("week")["dlog_sales"].transform("mean")
    )

    x = part["x"].to_numpy()
    y = part["y"].to_numpy()

    if len(part) < 100 or np.std(x) < 0.001:
        return None

    model = sm.OLS(
        y,
        sm.add_constant(x)
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": part["store"]}
    )

    ci = model.conf_int()[1]

    return {
        "period": label,
        "observations": len(part),
        "weeks": part["week"].nunique(),
        "stores": part["store"].nunique(),
        "coefficient": model.params[1],
        "clustered_se": model.bse[1],
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "r_squared": model.rsquared
    }

results = []

for label, part in [
    ("First half", df[df["week"] <= median_week]),
    ("Second half", df[df["week"] > median_week]),
    ("Full period", df)
]:
    result = fit_model(part, label)
    if result:
        results.append(result)

result_df = pd.DataFrame(results)
result_df.to_csv(out, index=False)

print("\n=== TEMPORAL STABILITY ===")
print(result_df.round(5).to_string(index=False))

first = result_df.loc[
    result_df["period"] == "First half", "coefficient"
].iloc[0]

second = result_df.loc[
    result_df["period"] == "Second half", "coefficient"
].iloc[0]

print("\nCoefficient difference (second - first):",
      round(second - first, 5))

print("\nSaved:", out)
print("TEMPORAL STABILITY ANALYSIS COMPLETED")
