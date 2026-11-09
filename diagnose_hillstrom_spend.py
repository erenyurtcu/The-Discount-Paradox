from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path("Hillstrom_Dataset")
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
].copy().reset_index(drop=True)

df["treatment"] = (
    df["segment"] == "Mens E-Mail"
).astype(int)

t = df["treatment"].to_numpy()
indices = np.arange(len(df))

development, test_idx = train_test_split(
    indices, test_size=0.20,
    stratify=t, random_state=42
)

development, val_idx = train_test_split(
    development, test_size=0.1875,
    stratify=t[development], random_state=43
)

nuisance_idx, cate_idx = train_test_split(
    development, test_size=25/65,
    stratify=t[development], random_state=44
)

splits = {
    "Nuisance": nuisance_idx,
    "CATE training": cate_idx,
    "Validation": val_idx
}

print("=== SPEND DISTRIBUTION AUDIT ===")

for name, idx in splits.items():
    part = df.iloc[idx]
    spend = part["spend"]

    print(f"\n{name}: {len(part):,} customers")
    print("Zero spend %:", round(100 * (spend == 0).mean(), 3))
    print("Mean spend:", round(spend.mean(), 5))
    print("Median spend:", round(spend.median(), 5))
    print("99th percentile:", round(spend.quantile(.99), 5))
    print("Maximum spend:", round(spend.max(), 5))

    group = part.groupby("segment")["spend"].agg(
        ["count", "mean", "median", "max"]
    )
    print(group.to_string())

    threshold = spend.quantile(.99)
    total = spend.sum()
    share = (
        spend[spend >= threshold].sum() / total
        if total > 0 else np.nan
    )

    print("Top 1% spend share:", round(share, 5))

print("\n=== VALIDATION ARM DIFFERENCE ===")
validation = df.iloc[val_idx]

treated = validation.loc[
    validation["treatment"] == 1, "spend"
]
control = validation.loc[
    validation["treatment"] == 0, "spend"
]

print("Treated mean:", round(treated.mean(), 6))
print("Control mean:", round(control.mean(), 6))
print("Observed ATE:", round(treated.mean() - control.mean(), 6))

print("\nHILLSTROM SPEND DIAGNOSTICS COMPLETED")
