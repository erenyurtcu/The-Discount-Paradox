from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm

root = Path("Hillstrom_Dataset")
processed = root / "Processed"

files = [
    p for p in root.rglob("*.csv")
    if "Processed" not in p.parts
    and "MineThatData" in p.name
]

if len(files) != 1:
    raise RuntimeError(f"Original CSV not unique: {files}")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.lower()

df = df[
    df["segment"].isin(["No E-Mail", "Mens E-Mail"])
].reset_index(drop=True)

t = (df["segment"] == "Mens E-Mail").astype(int).to_numpy()
y = df["spend"].to_numpy(dtype=float)

indices = np.arange(len(df))

development, _ = train_test_split(
    indices, test_size=0.20,
    stratify=t, random_state=42
)

development, validation = train_test_split(
    development, test_size=0.1875,
    stratify=t[development], random_state=43
)

hurdle = pd.read_parquet(
    processed / "hillstrom_hurdle_validation_predictions.parquet"
)

if len(hurdle) != len(validation):
    raise RuntimeError("Validation row count mismatch")

if not np.array_equal(
    hurdle["treatment"].to_numpy(), t[validation]
):
    raise RuntimeError("Validation treatment order mismatch")

if not np.allclose(
    hurdle["spend"].to_numpy(), y[validation]
):
    raise RuntimeError("Validation outcome order mismatch")

# Existing DR validation predictions were not saved.
# For a fair comparison, we must recover the DR model's
# exact validation rankings, not approximate them.
dr_predictions = processed / "hillstrom_dr_validation_predictions.parquet"

if not dr_predictions.exists():
    print("DR validation predictions are not saved.")
    print("Hurdle validation data verified.")
    print("Next step: reproduce DR rankings with identical splits.")
    print("COMPARISON PRECHECK COMPLETED")
    raise SystemExit(0)

dr = pd.read_parquet(dr_predictions)

if len(dr) != len(validation):
    raise RuntimeError("DR prediction count mismatch")

if not np.array_equal(
    dr["treatment"].to_numpy(), t[validation]
):
    raise RuntimeError("DR treatment order mismatch")

if not np.allclose(
    dr["spend"].to_numpy(), y[validation]
):
    raise RuntimeError("DR outcome order mismatch")

def top_policy(predictions, fraction):
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nonfinite CATE predictions")
    action = np.zeros(len(predictions))
    order = np.argsort(-predictions, kind="stable")
    action[order[:round(fraction * len(predictions))]] = 1
    return action

dr_action = top_policy(dr["predicted_cate"], 0.40)
hurdle_action = top_policy(hurdle["predicted_cate"], 0.20)

cost = 0.50
tt = t[validation]
yy = y[validation]
p = 0.5

contrast = tt * yy / p - (1 - tt) * yy / (1 - p)

difference = (dr_action - hurdle_action) * (contrast - cost)

estimate = difference.mean()
se = difference.std(ddof=1) / np.sqrt(len(difference))
low = estimate - 1.96 * se
high = estimate + 1.96 * se

print("=== EXPLORATORY PAIRED COMPARISON ===")
print("DR Top 40% minus Hurdle Top 20%:", round(estimate, 6))
print("95% CI:", (round(low, 6), round(high, 6)))
print("p-value:", round(2 * norm.sf(abs(estimate / se)), 6))
print("Same customers targeted:", int(((dr_action == 1) & (hurdle_action == 1)).sum()))
print("EXPLORATORY COMPARISON COMPLETED")
