from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

root = Path("Hillstrom_Dataset") / "Processed"

dr = pd.read_parquet(
    root / "hillstrom_dr_validation_predictions.parquet"
)

hurdle = pd.read_parquet(
    root / "hillstrom_hurdle_validation_predictions.parquet"
)

if len(dr) != len(hurdle):
    raise RuntimeError("Different validation sizes")

if not np.array_equal(
    dr["treatment"].to_numpy(),
    hurdle["treatment"].to_numpy()
):
    raise RuntimeError("Treatment order mismatch")

if not np.allclose(
    dr["spend"].to_numpy(),
    hurdle["spend"].to_numpy()
):
    raise RuntimeError("Outcome order mismatch")

a = dr["predicted_cate"].to_numpy()
b = hurdle["predicted_cate"].to_numpy()

if not np.isfinite(a).all() or not np.isfinite(b).all():
    raise RuntimeError("Nonfinite CATE predictions")

print("=== CATE RANKING AGREEMENT ===")
print("Spearman:", round(spearmanr(a, b).statistic, 6))
print("Kendall tau:", round(kendalltau(a, b).statistic, 6))

rows = []
n = len(a)

for fraction in [0.10, 0.20, 0.40, 0.60, 0.80]:
    k = round(fraction * n)

    dr_selected = set(np.argsort(-a)[:k])
    hurdle_selected = set(np.argsort(-b)[:k])

    intersection = len(dr_selected & hurdle_selected)
    union = len(dr_selected | hurdle_selected)

    rows.append({
        "target_fraction": fraction,
        "target_count_each": k,
        "shared_customers": intersection,
        "overlap_fraction": intersection / k,
        "jaccard": intersection / union
    })

result = pd.DataFrame(rows)

print("\n=== TOP-K TARGETING AGREEMENT ===")
print(result.round(5).to_string(index=False))

result.to_csv(
    root / "hillstrom_model_agreement.csv",
    index=False
)

print("\nHILLSTROM MODEL AGREEMENT COMPLETED")
