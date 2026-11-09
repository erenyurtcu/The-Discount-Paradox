from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

print("1/3 - Preparing price-demand panel...", flush=True)

df = con.execute("""
    SELECT
        store,
        week,
        move,
        price,
        qty,
        price / NULLIF(qty, 0) AS unit_price,
        sale
    FROM read_parquet(?)
    WHERE price > 0
      AND qty > 0
      AND move >= 0
""", [str(src)]).fetchdf()

df["log_sales"] = np.log1p(df["move"])
df["log_price"] = np.log(df["unit_price"])

print("Observations:", len(df))
print("Stores:", df["store"].nunique())
print("Weeks:", df["week"].nunique())

print("2/3 - Fitting fixed-effects regression...", flush=True)

model = smf.ols(
    "log_sales ~ log_price + C(store) + C(week)",
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["store"]}
)

beta = model.params["log_price"]
se = model.bse["log_price"]
ci = model.conf_int().loc["log_price"]

print("\n=== PRICE-DEMAND MODEL ===")
print("Coefficient:", round(beta, 5))
print("Clustered SE:", round(se, 5))
print("95% CI:", [round(float(x), 5) for x in ci])
print("P-value:", model.pvalues["log_price"])
print("R-squared:", round(model.rsquared, 5))

result = pd.DataFrame([{
    "product": "DIET COKE 2 LT",
    "upc": "4900000663",
    "observations": len(df),
    "stores": df["store"].nunique(),
    "weeks": df["week"].nunique(),
    "price_coefficient": beta,
    "clustered_se": se,
    "ci_lower": ci.iloc[0],
    "ci_upper": ci.iloc[1],
    "p_value": model.pvalues["log_price"],
    "r_squared": model.rsquared,
    "outcome_transform": "log1p(move)",
    "price_transform": "log(price/qty)",
    "causal_status": "observational_association"
}])

result.to_csv(out / "diet_coke_fixed_effects_results.csv", index=False)

print("\n3/3 - Saved model results")
print("DIET COKE FIXED EFFECTS MODEL COMPLETED")
