
from pathlib import Path
import pandas as pd
import numpy as np

root = Path.cwd()
files = list((root / "Hillstrom_Dataset").rglob("*.csv"))

if len(files) != 1:
    raise RuntimeError(f"Expected one CSV, found {len(files)}")

df = pd.read_csv(files[0])

groups = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
covariates = [
    "recency", "history", "mens", "womens", "newbie"
]

print("=== GROUP SIZES ===")
print(df["segment"].value_counts().to_string())

print("\n=== PRE-TREATMENT COVARIATE MEANS ===")
print(
    df.groupby("segment")[covariates]
    .mean().round(4).to_string()
)

print("\n=== STANDARDIZED MEAN DIFFERENCES ===")

control = df[df["segment"] == "No E-Mail"]

results = []

for group in groups[1:]:
    treated = df[df["segment"] == group]

    for variable in covariates:
        x = treated[variable].astype(float)
        y = control[variable].astype(float)

        pooled_sd = np.sqrt(
            (x.var(ddof=1) + y.var(ddof=1)) / 2
        )

        smd = (
            (x.mean() - y.mean()) / pooled_sd
            if pooled_sd > 0 else 0.0
        )

        results.append({
            "treatment": group,
            "covariate": variable,
            "smd": smd,
            "abs_smd": abs(smd)
        })

balance = pd.DataFrame(results)

print(balance.round(5).to_string(index=False))
print("\nMaximum absolute SMD:", balance["abs_smd"].max())

print("\n=== CATEGORICAL COVARIATE DISTRIBUTIONS ===")

for variable in ["history_segment", "zip_code", "channel"]:
    print(f"\n{variable}")
    print(
        pd.crosstab(
            df["segment"],
            df[variable],
            normalize="index"
        ).round(4).to_string()
    )

print("\n=== OBSERVED TREATMENT EFFECTS ===")

outcomes = ["visit", "conversion", "spend"]
means = df.groupby("segment")[outcomes].mean()

for group in groups[1:]:
    print(f"\n{group} vs No E-Mail")
    for outcome in outcomes:
        difference = (
            means.loc[group, outcome]
            - means.loc["No E-Mail", outcome]
        )
        print(f"{outcome}: {difference:.6f}")

output = root / "Dataset_Reports"
output.mkdir(exist_ok=True)

balance.to_csv(
    output / "hillstrom_balance.csv",
    index=False
)

print("\nHILLSTROM BALANCE AUDIT COMPLETED")
