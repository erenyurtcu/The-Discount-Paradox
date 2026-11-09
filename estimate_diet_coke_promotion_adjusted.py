from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_promotion_adjusted_results.csv")

con = duckdb.connect()

df = con.execute("""
WITH base AS (
    SELECT
        store,
        week,
        LN(price / qty) AS log_price,
        LN(move) AS log_sales,
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
        ) AS prev_week,
        LAG(log_price) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_log_price,
        LAG(log_sales) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_log_sales,
        LAG(sale_code) OVER (
            PARTITION BY store ORDER BY week
        ) AS prev_sale_code
    FROM base
)
SELECT
    store,
    week,
    log_price - prev_log_price AS dlog_price,
    log_sales - prev_log_sales AS dlog_sales,
    sale_code,
    prev_sale_code
FROM lagged
WHERE prev_week = week - 1
""", [str(src)]).fetchdf()

con.close()

print("Observations:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

for col in ["dlog_price", "dlog_sales"]:
    df[col + "_adj"] = (
        df[col] - df.groupby("week")[col].transform("mean")
    )

def demean_week(values):
    return values - values.groupby(df["week"]).transform("mean")

def estimate(name, include_promotion):
    X = pd.DataFrame({
        "price": df["dlog_price_adj"]
    })

    if include_promotion:
        for code in ["S", "B"]:
            current = (df["sale_code"] == code).astype(float)
            previous = (df["prev_sale_code"] == code).astype(float)
            X["delta_promo_" + code] = demean_week(current - previous)

    y = df["dlog_sales_adj"]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["store"]}
    )

    ci = model.conf_int().loc["price"]

    return {
        "model": name,
        "observations": len(df),
        "price_coefficient": model.params["price"],
        "clustered_se": model.bse["price"],
        "ci_lower": ci.iloc[0],
        "ci_upper": ci.iloc[1],
        "p_value": model.pvalues["price"],
        "r_squared": model.rsquared,
        "promo_S_coefficient": model.params.get("delta_promo_S", np.nan),
        "promo_B_coefficient": model.params.get("delta_promo_B", np.nan)
    }

results = [
    estimate("Week-adjusted", False),
    estimate("Week + promotion-adjusted", True)
]

result = pd.DataFrame(results)
result.to_csv(out, index=False)

print("\n=== PROMOTION-ADJUSTED COMPARISON ===")
print(result.round(5).to_string(index=False))

print("\nSaved:", out)
print("PROMOTION-ADJUSTED ANALYSIS COMPLETED")
