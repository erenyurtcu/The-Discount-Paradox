from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import norm

ROOT = Path("Hillstrom_Dataset")
OUT = ROOT / "Processed"
OUT.mkdir(parents=True, exist_ok=True)

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
conversion = df["conversion"].to_numpy(dtype=int)

idx = np.arange(len(df))

development, unused_test = train_test_split(
    idx, test_size=0.20,
    stratify=t, random_state=42
)

development, validation = train_test_split(
    development, test_size=0.1875,
    stratify=t[development], random_state=43
)

print("=== DATA PARTITIONS ===")
print("Development:", len(development))
print("Validation:", len(validation))
print("Previously inspected test excluded:", len(unused_test))

models = {}

for arm in [0, 1]:
    train_idx = development[t[development] == arm]
    positive_idx = train_idx[conversion[train_idx] == 1]

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=3000,
            class_weight=None,
            random_state=42
        )
    )

    classifier.fit(
        X.iloc[train_idx],
        conversion[train_idx]
    )

    # Fit log-spend among converters, with regularization.
    regressor = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0)
    )

    regressor.fit(
        X.iloc[positive_idx],
        np.log(y[positive_idx])
    )

    # Smearing correction on positive training observations.
    residuals = (
        np.log(y[positive_idx])
        - regressor.predict(X.iloc[positive_idx])
    )
    smearing = np.exp(residuals).mean()

    models[arm] = (classifier, regressor, smearing)

    print(
        f"Arm {arm}: {len(train_idx):,} customers, "
        f"{len(positive_idx)} positive spend observations"
    )

def expected_spend(arm, data):
    classifier, regressor, smearing = models[arm]
    probability = classifier.predict_proba(data)[:, 1]
    conditional_spend = (
        np.exp(
            np.clip(regressor.predict(data), -20, 20)
        ) * smearing
    )
    return probability * conditional_spend

X_val = X.iloc[validation]

mu0 = expected_spend(0, X_val)
mu1 = expected_spend(1, X_val)
cate = mu1 - mu0

observed_t = t[validation]
observed_y = y[validation]

# Evaluation uses known two-arm randomized assignment.
p = 0.5
contrast = (
    observed_t * observed_y / p
    - (1 - observed_t) * observed_y / (1 - p)
)

ranking = np.argsort(-cate, kind="stable")

policies = {
    "None": np.zeros(len(validation), dtype=int),
    "Everyone": np.ones(len(validation), dtype=int)
}

for fraction in [0.2, 0.4, 0.6, 0.8]:
    action = np.zeros(len(validation), dtype=int)
    n_selected = round(fraction * len(validation))
    action[ranking[:n_selected]] = 1
    policies[f"Top {int(fraction*100)}%"] = action

COST = 0.50
rows = []

for name, action in policies.items():
    values = action * (contrast - COST)
    estimate = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))

    rows.append({
        "policy": name,
        "target_fraction": action.mean(),
        "net_value": estimate,
        "ci95_low": estimate - 1.96 * se,
        "ci95_high": estimate + 1.96 * se
    })

result = pd.DataFrame(rows)

print("\n=== HURDLE T-LEARNER VALIDATION ===")
print(result.round(5).to_string(index=False))

print("\n=== MODEL DIAGNOSTICS ===")
print("Mean predicted mu0:", round(mu0.mean(), 6))
print("Mean predicted mu1:", round(mu1.mean(), 6))
print("Mean predicted CATE:", round(cate.mean(), 6))
print("Observed validation ATE:", round(contrast.mean(), 6))

result.to_csv(
    OUT / "hillstrom_hurdle_validation.csv",
    index=False
)

pd.DataFrame({
    "predicted_cate": cate,
    "treatment": observed_t,
    "spend": observed_y
}).to_parquet(
    OUT / "hillstrom_hurdle_validation_predictions.parquet",
    index=False
)

print("\nHILLSTROM HURDLE T-LEARNER COMPLETED")
