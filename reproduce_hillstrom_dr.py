from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ROOT = Path("Hillstrom_Dataset")
OUT = ROOT / "Processed"

files = [
    p for p in ROOT.rglob("*.csv")
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

features = [
    "recency", "history", "mens", "womens",
    "newbie", "history_segment", "zip_code", "channel"
]

X = pd.get_dummies(
    df[features],
    columns=["history_segment", "zip_code", "channel"],
    dtype=float
).fillna(0)

t = (df["segment"] == "Mens E-Mail").astype(int).to_numpy()
y = df["spend"].to_numpy(dtype=float)

indices = np.arange(len(df))

development, unused_test = train_test_split(
    indices,
    test_size=0.20,
    stratify=t,
    random_state=42
)

development, validation = train_test_split(
    development,
    test_size=0.1875,
    stratify=t[development],
    random_state=43
)

nuisance_idx, cate_idx = train_test_split(
    development,
    test_size=25/65,
    stratify=t[development],
    random_state=44
)

print("=== REPRODUCING DR-LEARNER ===", flush=True)
print("Nuisance:", len(nuisance_idx))
print("CATE training:", len(cate_idx))
print("Validation:", len(validation))
print("Excluded prior test:", len(unused_test))

def make_forest(seed):
    return RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=80,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed
    )

mu = {}

for arm in [0, 1]:
    arm_idx = nuisance_idx[t[nuisance_idx] == arm]

    model = make_forest(42 + arm)
    model.fit(X.iloc[arm_idx], y[arm_idx])

    mu[arm] = model
    print(f"Nuisance arm {arm} completed", flush=True)

m0 = mu[0].predict(X.iloc[cate_idx])
m1 = mu[1].predict(X.iloc[cate_idx])

tt = t[cate_idx]
yy = y[cate_idx]
p = 0.5

pseudo = (
    m1 - m0
    + tt * (yy - m1) / p
    - (1 - tt) * (yy - m0) / (1 - p)
)

dr_model = make_forest(52)
dr_model.fit(X.iloc[cate_idx], pseudo)

predicted_cate = dr_model.predict(X.iloc[validation])

result = pd.DataFrame({
    "predicted_cate": predicted_cate,
    "treatment": t[validation],
    "spend": y[validation]
})

# Verify reproduced rankings against previously saved
# aggregate validation policy values.
ranking = np.argsort(-predicted_cate, kind="stable")
contrast = (
    t[validation] * y[validation] / p
    - (1 - t[validation]) * y[validation] / (1-p)
)

action = np.zeros(len(validation))
action[ranking[:round(0.40 * len(validation))]] = 1

reproduced_value = np.mean(action * (contrast - 0.50))

reference = pd.read_csv(
    OUT / "hillstrom_dr_validation.csv"
)

reference_value = float(
    reference.loc[
        reference["policy"] == "Top 40%",
        "net_value"
    ].iloc[0]
)

print("\n=== REPRODUCTION CHECK ===")
print("Previous Top 40% value:", round(reference_value, 8))
print("Reproduced Top 40% value:", round(reproduced_value, 8))
print("Absolute difference:", round(abs(reference_value - reproduced_value), 10))

if not np.isclose(
    reproduced_value,
    reference_value,
    atol=1e-6,
    rtol=0
):
    raise RuntimeError(
        "DR reproduction mismatch. Predictions NOT saved."
    )

path = OUT / "hillstrom_dr_validation_predictions.parquet"
result.to_parquet(path, index=False)

print("\nSaved:", path)
print("DR VALIDATION PREDICTIONS REPRODUCED")
