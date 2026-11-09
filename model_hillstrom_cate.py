from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

root = Path("Hillstrom_Dataset")
files = [
    p for p in root.rglob("*.csv")
    if "Processed" not in p.parts
    and "MineThatData" in p.name
]
if len(files) != 1:
    raise RuntimeError(f"Expected one original Hillstrom CSV, found {len(files)}: {files}")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.lower()

features = [
    "recency", "history", "mens", "womens",
    "newbie", "history_segment", "zip_code", "channel"
]

required = features + ["segment", "spend"]
if any(c not in df.columns for c in required):
    raise RuntimeError(
        f"Missing columns: {set(required) - set(df.columns)}"
    )

df = df[
    df["segment"].isin(["No E-Mail", "Mens E-Mail"])
].copy()

df["treatment"] = (
    df["segment"] == "Mens E-Mail"
).astype(int)

X = pd.get_dummies(
    df[features],
    columns=["history_segment", "zip_code", "channel"],
    dtype=float
).fillna(0)

y = df["spend"].astype(float)
t = df["treatment"]

X_train, X_test, y_train, y_test, t_train, t_test = (
    train_test_split(
        X, y, t,
        test_size=0.30,
        random_state=42,
        stratify=t
    )
)

models = {}

for group, name in [(0, "control"), (1, "treated")]:
    mask = t_train == group

    model = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=100,
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train.loc[mask], y_train.loc[mask])
    models[group] = model

    print(f"{name} model trained: {mask.sum():,} rows")

mu0 = models[0].predict(X_test)
mu1 = models[1].predict(X_test)
cate = mu1 - mu0

test = pd.DataFrame({
    "predicted_cate": cate,
    "treatment": t_test.to_numpy(),
    "spend": y_test.to_numpy()
})

test["quintile"] = pd.qcut(
    test["predicted_cate"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rows = []

for q, group in test.groupby("quintile"):
    treated = group.loc[group.treatment == 1, "spend"]
    control = group.loc[group.treatment == 0, "spend"]

    observed_ate = treated.mean() - control.mean()

    se = np.sqrt(
        treated.var(ddof=1) / len(treated)
        + control.var(ddof=1) / len(control)
    )

    rows.append({
        "quintile": q,
        "n": len(group),
        "n_treated": len(treated),
        "n_control": len(control),
        "predicted_cate_mean": group.predicted_cate.mean(),
        "observed_ate": observed_ate,
        "ci95_low": observed_ate - 1.96 * se,
        "ci95_high": observed_ate + 1.96 * se
    })

result = pd.DataFrame(rows)

output_dir = root / "Processed"
output_dir.mkdir(parents=True, exist_ok=True)

result.to_csv(
    output_dir / "hillstrom_cate_quintiles.csv",
    index=False
)

test.to_parquet(
    output_dir / "hillstrom_cate_holdout.parquet",
    index=False
)

print("\n=== OUT-OF-SAMPLE CATE VALIDATION ===")
print(result.round(4).to_string(index=False))

print("\nOverall holdout observed ATE:",
      round(
          test.loc[test.treatment == 1, "spend"].mean()
          - test.loc[test.treatment == 0, "spend"].mean(),
          6
      ))

print("Mean predicted CATE:", round(cate.mean(), 6))
print("\nHILLSTROM CATE BASELINE COMPLETED")
