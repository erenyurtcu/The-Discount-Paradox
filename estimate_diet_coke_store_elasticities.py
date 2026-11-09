from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

source = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
output = Path("Dominicks_Dataset/Analysis/diet_coke_store_elasticities.csv")

con = duckdb.connect()
df = con.execute("""
SELECT
    store,
    week,
    move,
    price / qty AS unit_price
FROM read_parquet(?)
WHERE price > 0
  AND qty > 0
  AND move > 0
ORDER BY store, week
""", [str(source)]).fetchdf()
con.close()

df["log_price"] = np.log(df["unit_price"])
df["log_units"] = np.log(df["move"])

grouped = df.groupby("store", sort=False)
df["prev_week"] = grouped["week"].shift()
df["dlog_price"] = grouped["log_price"].diff()
df["dlog_units"] = grouped["log_units"].diff()

pairs = df.loc[
    (df["week"] - df["prev_week"] == 1)
    & np.isfinite(df["dlog_price"])
    & np.isfinite(df["dlog_units"])
].copy()

results = []

for store, part in pairs.groupby("store"):
    if len(part) < 50:
        continue

    x = part["dlog_price"].to_numpy()
    y = part["dlog_units"].to_numpy()

    if np.std(x) < 0.01:
        continue

    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]

    results.append({
        "store": store,
        "week_pairs": len(part),
        "elasticity_association": slope,
        "correlation": correlation,
        "price_change_std": np.std(x)
    })

result = pd.DataFrame(results)
result.to_csv(output, index=False)

print("=== STORE-LEVEL PRICE ASSOCIATIONS ===")
print("Stores modeled:", len(result))
print(result["elasticity_association"].describe().round(3).to_string())

print("\nNegative associations:",
      int((result["elasticity_association"] < 0).sum()))
print("Positive associations:",
      int((result["elasticity_association"] > 0).sum()))

print("\nFive most negative:")
print(result.nsmallest(
    5, "elasticity_association"
).to_string(index=False))

print("\nFive most positive:")
print(result.nlargest(
    5, "elasticity_association"
).to_string(index=False))

print("\nSaved:", output)
print("STORE ELASTICITY ANALYSIS COMPLETED")
