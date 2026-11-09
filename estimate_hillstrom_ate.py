from pathlib import Path
import pandas as pd
import numpy as np
from math import erfc, sqrt

root = Path("Hillstrom_Dataset")
files = sorted(p for p in root.glob("*.csv") if "MineThatData" in p.name and p.is_file())

if len(files) != 1:
    raise RuntimeError(
        f"Expected exactly one raw Hillstrom CSV in {root}, found {len(files)}: {files}"
    )

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.lower()

required = ["segment", "visit", "conversion", "spend"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

assert len(df) == 64000
assert not df[required].isna().any().any()

control_name = "No E-Mail"
treatments = ["Mens E-Mail", "Womens E-Mail"]
expected_groups = {control_name, *treatments}

if set(df["segment"].unique()) != expected_groups:
    raise RuntimeError(
        f"Unexpected groups: {df['segment'].unique()}"
    )

control = df[df["segment"] == control_name]
results = []

for treatment in treatments:
    treated = df[df["segment"] == treatment]

    for outcome in ["visit", "conversion", "spend"]:
        y1 = treated[outcome].astype(float)
        y0 = control[outcome].astype(float)

        mean1 = y1.mean()
        mean0 = y0.mean()
        ate = mean1 - mean0

        se = np.sqrt(
            y1.var(ddof=1) / len(y1)
            + y0.var(ddof=1) / len(y0)
        )

        ci_low = ate - 1.96 * se
        ci_high = ate + 1.96 * se

        z = ate / se if se > 0 else np.nan
        p_value = (
            erfc(abs(z) / sqrt(2))
            if np.isfinite(z) else np.nan
        )

        results.append({
            "treatment": treatment,
            "control": control_name,
            "outcome": outcome,
            "n_treated": len(y1),
            "n_control": len(y0),
            "treated_mean": mean1,
            "control_mean": mean0,
            "ate": ate,
            "standard_error": se,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "p_value_normal": p_value
        })

result = pd.DataFrame(results)

output_dir = root / "Processed"
output_dir.mkdir(parents=True, exist_ok=True)

output = output_dir / "hillstrom_experimental_ate.csv"
result.to_csv(output, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("=== HILLSTROM EXPERIMENTAL ATE ===")
print(result.to_string(index=False))

print("\nRandomization groups:")
print(df["segment"].value_counts().to_string())

print("\nSaved:", output)
print("HILLSTROM EXPERIMENTAL ATE COMPLETED")
