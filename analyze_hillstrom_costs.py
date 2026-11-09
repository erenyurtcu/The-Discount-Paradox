from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

root = Path("Hillstrom_Dataset") / "Processed"
df = pd.read_parquet(root / "hillstrom_cate_holdout.parquet")

t = df["treatment"].to_numpy(dtype=int)
y = df["spend"].to_numpy(dtype=float)
q = df["quintile"].to_numpy(dtype=int)

if not np.isin(t, [0, 1]).all():
    raise RuntimeError("Invalid treatment labels")

# Empirical allocation probability in this holdout.
p = t.mean()

# Randomized experiment: incremental outcome contribution.
contrast = t * y / p - (1 - t) * y / (1 - p)

policies = {
    "None": np.zeros(len(df)),
    "Top 20%": (q == 5).astype(float),
    "Top 40%": (q >= 4).astype(float),
    "Top 60%": (q >= 3).astype(float),
    "Top 80%": (q >= 2).astype(float),
    "Everyone": np.ones(len(df))
}

costs = [0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
rows = []

for cost in costs:
    for name, policy in policies.items():
        contribution = policy * (contrast - cost)

        estimate = contribution.mean()
        se = contribution.std(ddof=1) / np.sqrt(len(df))

        rows.append({
            "cost_per_email": cost,
            "policy": name,
            "target_fraction": policy.mean(),
            "net_incremental_value": estimate,
            "ci95_low": estimate - 1.96 * se,
            "ci95_high": estimate + 1.96 * se
        })

results = pd.DataFrame(rows)

print("=== COST SENSITIVITY ===")
for cost, group in results.groupby("cost_per_email"):
    winner = group.loc[group["net_incremental_value"].idxmax()]
    print(
        f"Cost={cost:.2f} | "
        f"Best point estimate={winner['policy']} | "
        f"Net value={winner['net_incremental_value']:.4f}"
    )

# Paired comparison of each targeting policy vs everyone.
comparisons = []

for name, policy in policies.items():
    if name == "Everyone":
        continue

    # No campaign cost in this first comparison.
    difference = (policy - 1) * contrast

    estimate = difference.mean()
    se = difference.std(ddof=1) / np.sqrt(len(df))
    p_value = (
        2 * norm.sf(abs(estimate / se))
        if se > 0 else np.nan
    )

    comparisons.append({
        "policy": name,
        "difference_vs_everyone": estimate,
        "ci95_low": estimate - 1.96 * se,
        "ci95_high": estimate + 1.96 * se,
        "p_value": p_value
    })

comparison_df = pd.DataFrame(comparisons)

print("\n=== PAIRED POLICY COMPARISONS (ZERO COST) ===")
print(comparison_df.round(5).to_string(index=False))

results.to_csv(
    root / "hillstrom_cost_sensitivity.csv",
    index=False
)

comparison_df.to_csv(
    root / "hillstrom_paired_policy_comparisons.csv",
    index=False
)

print("\nHILLSTROM COST SENSITIVITY COMPLETED")
