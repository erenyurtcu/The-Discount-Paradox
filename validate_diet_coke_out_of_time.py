from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

src = Path("Dominicks_Dataset/Analysis/sdr_pilot_4900000663.parquet")
out = Path("Dominicks_Dataset/Analysis/diet_coke_out_of_time_validation.csv")

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

for col in ["dlog_price", "dlog_sales"]:
    df[col + "_adj"] = (
        df[col] - df.groupby("week")[col].transform("mean")
    )

split = 195
train = df[df["week"] <= split].copy()
test = df[df["week"] > split].copy()

x_train = train["dlog_price_adj"].to_numpy()
y_train = train["dlog_sales_adj"].to_numpy()

beta = np.dot(x_train, y_train) / np.dot(x_train, x_train)

x_test = test["dlog_price_adj"].to_numpy()
y_test = test["dlog_sales_adj"].to_numpy()

pred_model = beta * x_test
pred_baseline = np.zeros_like(y_test)

def metrics(name, predictions):
    errors = y_test - predictions
    return {
        "model": name,
        "test_observations": len(y_test),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "mae": np.mean(np.abs(errors))
    }

results = pd.DataFrame([
    metrics("Zero-change baseline", pred_baseline),
    metrics("Price association model", pred_model)
])

baseline_rmse = results.loc[0, "rmse"]
model_rmse = results.loc[1, "rmse"]

print("=== OUT-OF-TIME DIAGNOSTIC ===")
print("Train observations:", len(train))
print("Test observations:", len(test))
print("Train coefficient:", round(beta, 5))
print(results.round(5).to_string(index=False))

print(
    "\nRMSE improvement (%):",
    round(100 * (baseline_rmse - model_rmse) / baseline_rmse, 2)
)

results.to_csv(out, index=False)

print("\nSaved:", out)
print("OUT-OF-TIME DIAGNOSTIC COMPLETED")
