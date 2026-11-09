from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

root = Path("Hillstrom_Dataset") / "Processed"
source = root / "hillstrom_cate_holdout.parquet"

df = pd.read_parquet(source)

required = ["predicted_cate", "treatment", "spend", "quintile"]
if not set(required).issubset(df.columns):
    raise RuntimeError("Missing required holdout columns")

if df[required].isna().any().any():
    raise RuntimeError("Missing values in holdout data")

if not df["treatment"].isin([0, 1]).all():
    raise RuntimeError("Invalid treatment values")

n = len(df)
t = df["treatment"].to_numpy(dtype=int)
y = df["spend"].to_numpy(dtype=float)
q = df["quintile"].to_numpy(dtype=int)

# Known randomized assignment probability.
# Mens E-Mail and No E-Mail were approximately equally allocated.
p = 0.5

def estimate(values):
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se

# Incremental spend from a policy, relative to no email.
# This is NOT net profit: campaign costs are unavailable.
treatment_contrast = t * y / p - (1 - t) * y / (1 - p)

policies = {
    "No email": np.zeros(n),
    "Email everyone": np.ones(n),
    "Email predicted top 20%": (q == 5).astype(float),
    "Email predicted top 40%": (q >= 4).astype(float),
    "Email predicted top 60%": (q >= 3).astype(float),
    "Email predicted top 80%": (q >= 2).astype(float)
}

rows = []

for name, policy in policies.items():
    contribution = policy * treatment_contrast
    mean, se, low, high = estimate(contribution)

    rows.append({
        "policy": name,
        "target_fraction": policy.mean(),
        "incremental_spend_per_customer": mean,
        "standard_error": se,
        "ci95_low": low,
        "ci95_high": high
    })

results = pd.DataFrame(rows)

# Difference between the observed treatment effects
# in the highest and lowest predicted CATE quintiles.
def subgroup_ate(group):
    yt = group.loc[group.treatment == 1, "spend"]
    yc = group.loc[group.treatment == 0, "spend"]

    ate = yt.mean() - yc.mean()
    variance = (
        yt.var(ddof=1) / len(yt)
        + yc.var(ddof=1) / len(yc)
    )
    return ate, variance

q1 = df[df.quintile == 1]
q5 = df[df.quintile == 5]

ate1, var1 = subgroup_ate(q1)
ate5, var5 = subgroup_ate(q5)

difference = ate5 - ate1
se_difference = np.sqrt(var1 + var5)
ci_low = difference - 1.96 * se_difference
ci_high = difference + 1.96 * se_difference
p_value = 2 * norm.sf(abs(difference / se_difference))

print("=== OUT-OF-SAMPLE POLICY EVALUATION ===")
print(results.round(5).to_string(index=False))

print("\n=== Q5 VS Q1 HETEROGENEITY TEST ===")
print(f"Q1 observed ATE: {ate1:.6f}")
print(f"Q5 observed ATE: {ate5:.6f}")
print(f"Difference Q5-Q1: {difference:.6f}")
print(f"Standard error: {se_difference:.6f}")
print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
print(f"Two-sided p-value: {p_value:.8f}")

output = root / "hillstrom_policy_evaluation.csv"
results.to_csv(output, index=False)

print("\nSaved:", output)
print("HILLSTROM POLICY EVALUATION COMPLETED")
