from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Preparing consecutive-week price panel...", flush=True)

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        move,
        price,
        qty,
        CASE
            WHEN price > 0 AND qty > 0
            THEN price / qty
            ELSE NULL
        END AS unit_price
    FROM read_parquet(?)
),
panel AS (
    SELECT
        *,
        LEAD(week) OVER (
            PARTITION BY store ORDER BY week
        ) AS next_week,
        LEAD(unit_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS next_price
    FROM base
)
SELECT
    store,
    week,
    move,
    unit_price,
    next_price
FROM panel
WHERE next_week = week + 1
  AND unit_price > 0
  AND next_price > 0
  AND move >= 0
""", [str(src)]).fetchdf()

df["log_sales"] = np.log1p(df["move"])
df["log_price"] = np.log(df["unit_price"])
df["log_future_price"] = np.log(df["next_price"])

print("Observations:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

print("2/3 - Fitting diagnostic models...", flush=True)

specs = [
    (
        "Current_price_only",
        "log_sales ~ log_price + C(store) + C(week)"
    ),
    (
        "Current_and_future_price",
        "log_sales ~ log_price + log_future_price"
        " + C(store) + C(week)"
    )
]

rows = []

for name, formula in specs:
    print("Fitting:", name, flush=True)

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store"]}
    )

    for variable in ["log_price", "log_future_price"]:
        if variable not in model.params:
            continue

        ci = model.conf_int().loc[variable]

        rows.append({
            "model": name,
            "variable": variable,
            "observations": len(df),
            "coefficient": model.params[variable],
            "clustered_se": model.bse[variable],
            "ci_lower": ci.iloc[0],
            "ci_upper": ci.iloc[1],
            "p_value": model.pvalues[variable],
            "r_squared": model.rsquared
        })

result = pd.DataFrame(rows)

print("\n=== FUTURE PRICE DIAGNOSTIC ===")
print(result.round(5).to_string(index=False))

result.to_csv(
    out / "diet_coke_future_price_diagnostic.csv",
    index=False
)

print("\n3/3 - Results saved")
print("DIET COKE FUTURE PRICE DIAGNOSTIC COMPLETED")
