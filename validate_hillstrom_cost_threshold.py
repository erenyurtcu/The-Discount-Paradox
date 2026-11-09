from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

root = Path("Hillstrom_Dataset") / "Processed"
df = pd.read_parquet(root / "hillstrom_cate_holdout.parquet")

t = df["treatment"].to_numpy(dtype=int)
y = df["spend"].to_numpy(dtype=float)
q = df["quintile"].to_numpy(dtype=int)

p = t.mean()
contrast = t * y / p - (1 - t) * y / (1 - p)

top80 = (q >= 2).astype(float)
everyone = np.ones(len(df))

# Positive values favor Top 80%.
excluded = everyone - top80
excluded_fraction = excluded.mean()

# Incremental value lost by excluding the bottom 20%.
lost_value = np.mean(excluded * contrast)

# Break-even campaign cost.
threshold = lost_value / excluded_fraction

print("=== TOP 80% VS EVERYONE ===")
print(f"Excluded fraction: {excluded_fraction:.6f}")
print(f"Lost incremental spend: {lost_value:.6f}")
print(f"Break-even cost: {threshold:.6f}")

rows = []

for cost in [0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50]:
    paired_difference = excluded * (cost - contrast)

    estimate = paired_difference.mean()
    se = paired_difference.std(ddof=1) / np.sqrt(len(df))

    low = estimate - 1.96 * se
    high = estimate + 1.96 * se
    p_value = (
        2 * norm.sf(abs(estimate / se))
        if se > 0 else np.nan
    )

    rows.append({
        "cost": cost,
        "top80_minus_everyone": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "p_value": p_value,
        "significant_top80_advantage": low > 0
    })

result = pd.DataFrame(rows)

print("\n=== PAIRED COST-ADJUSTED COMPARISON ===")
print(result.round(6).to_string(index=False))

# Bootstrap uncertainty for the break-even threshold.
# Resample evaluation rows, not training data.
rng = np.random.default_rng(42)
n = len(df)
bootstrap_thresholds = []

for _ in range(2000):
    idx = rng.integers(0, n, size=n)
    sampled_excluded = excluded[idx]
    fraction = sampled_excluded.mean()

    if fraction > 0:
        sampled_lost = (
            sampled_excluded * contrast[idx]
        ).mean()

        bootstrap_thresholds.append(
            sampled_lost / fraction
        )

bootstrap_ci = np.percentile(
    bootstrap_thresholds, [2.5, 97.5]
)

print("\n=== BREAK-EVEN THRESHOLD ===")
print(f"Point estimate: {threshold:.6f}")
print(
    f"Bootstrap 95% CI: "
    f"[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]"
)

result.to_csv(
    root / "hillstrom_cost_adjusted_comparison.csv",
    index=False
)

pd.DataFrame({
    "threshold": bootstrap_thresholds
}).to_csv(
    root / "hillstrom_threshold_bootstrap.csv",
    index=False
)

print("\nHILLSTROM COST THRESHOLD VALIDATION COMPLETED")
