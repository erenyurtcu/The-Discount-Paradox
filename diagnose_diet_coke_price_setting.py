from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Preparing historical demand and price changes...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        move,
        qty,
        CASE WHEN price > 0 AND qty > 0
             THEN price / qty
             ELSE NULL
        END AS unit_price
    FROM read_parquet(?)
),
lags AS (
    SELECT
        *,
        LAG(week, 1) OVER w AS week_lag1,
        LAG(week, 2) OVER w AS week_lag2,
        LAG(move, 1) OVER w AS move_lag1,
        LAG(move, 2) OVER w AS move_lag2,
        LAG(unit_price, 1) OVER w AS price_lag1,
        LAG(qty, 1) OVER w AS qty_lag1
    FROM base
    WINDOW w AS (PARTITION BY store ORDER BY week)
)
SELECT *
FROM lags
WHERE week_lag1 = week - 1
  AND week_lag2 = week - 2
  AND unit_price > 0
  AND price_lag1 > 0
  AND qty = qty_lag1
  AND move_lag1 >= 0
  AND move_lag2 >= 0
""", [str(src)]).fetchdf()

df["price_change"] = np.log(df["unit_price"]) - np.log(df["price_lag1"])
df["previous_sales_change"] = (
    np.log1p(df["move_lag1"]) - np.log1p(df["move_lag2"])
)

print("Observations:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

print("\n2/3 - Testing price-setting response...", flush=True)

model = smf.ols(
    "price_change ~ previous_sales_change + C(store) + C(week)",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["store"]}
)

variable = "previous_sales_change"
ci = model.conf_int().loc[variable]

result = pd.DataFrame([{
    "observations": len(df),
    "coefficient": model.params[variable],
    "clustered_se": model.bse[variable],
    "ci_lower": ci.iloc[0],
    "ci_upper": ci.iloc[1],
    "p_value": model.pvalues[variable],
    "r_squared": model.rsquared
}])

print("\n=== PRICE SETTING ENDOGENEITY DIAGNOSTIC ===")
print(result.round(6).to_string(index=False))

print("\n3/3 - Saving results...", flush=True)

result.to_csv(
    out / "diet_coke_price_setting_diagnostic.csv",
    index=False
)

print("DIET COKE PRICE SETTING DIAGNOSTIC COMPLETED")
