from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

df = con.execute("""
    SELECT store, week, move, price, qty, sale
    FROM read_parquet(?)
    WHERE price > 0 AND qty > 0 AND move >= 0
""", [str(src)]).fetchdf()

df["log_sales"] = np.log1p(df["move"])
df["log_price"] = np.log(df["price"] / df["qty"])

df["promotion"] = (
    df["sale"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
    .replace("", "NONE")
)

print("=== PROMOTION CODE DISTRIBUTION ===")
print(df["promotion"].value_counts().to_string())

specifications = [
    (
        "M1_baseline",
        df,
        "log_sales ~ log_price + C(store) + C(week)"
    ),
    (
        "M2_promotion",
        df,
        "log_sales ~ log_price + C(promotion) + C(store) + C(week)"
    ),
    (
        "M3_qty1",
        df.loc[df["qty"] == 1].copy(),
        "log_sales ~ log_price + C(promotion) + C(store) + C(week)"
    )
]

results = []

for name, sample, formula in specifications:
    print(f"\nFitting {name} ({len(sample):,} rows)...", flush=True)

    fitted = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["store"]}
    )

    ci = fitted.conf_int().loc["log_price"]

    results.append({
        "model": name,
        "observations": len(sample),
        "coefficient": fitted.params["log_price"],
        "clustered_se": fitted.bse["log_price"],
        "ci_lower": ci.iloc[0],
        "ci_upper": ci.iloc[1],
        "p_value": fitted.pvalues["log_price"],
        "r_squared": fitted.rsquared
    })

result = pd.DataFrame(results)

print("\n=== MODEL ROBUSTNESS COMPARISON ===")
print(result.round(5).to_string(index=False))

result.to_csv(
    out / "diet_coke_model_robustness.csv",
    index=False
)

print("\nDIET COKE MODEL ROBUSTNESS COMPLETED")
