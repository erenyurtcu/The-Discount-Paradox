from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ROOT = Path("Hillstrom_Dataset")
OUTPUT = ROOT / "Processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

COST = 0.50
SEED = 42

files = [
    p for p in ROOT.rglob("*.csv")
    if "Processed" not in p.parts
    and "MineThatData" in p.name
]

if len(files) != 1:
    raise RuntimeError(f"Original dataset not unique: {files}")

df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.lower()

features = [
    "recency", "history", "mens", "womens",
    "newbie", "history_segment", "zip_code", "channel"
]

df = df[
    df["segment"].isin(["No E-Mail", "Mens E-Mail"])
].copy().reset_index(drop=True)

df["treatment"] = (
    df["segment"] == "Mens E-Mail"
).astype(int)

X = pd.get_dummies(
    df[features],
    columns=["history_segment", "zip_code", "channel"],
    dtype=float
).fillna(0)

y = df["spend"].to_numpy(dtype=float)
t = df["treatment"].to_numpy(dtype=int)

indices = np.arange(len(df))

# 20% untouched final test.
development, test_idx = train_test_split(
    indices, test_size=0.20,
    stratify=t, random_state=SEED
)

# 15% validation, 65% model development.
development, val_idx = train_test_split(
    development, test_size=0.1875,
    stratify=t[development], random_state=SEED + 1
)

# 40% nuisance, 25% CATE training.
nuisance_idx, cate_idx = train_test_split(
    development, test_size=25/65,
    stratify=t[development], random_state=SEED + 2
)

print("=== DATA SPLIT ===", flush=True)
for name, idx in [
    ("Nuisance", nuisance_idx),
    ("CATE", cate_idx),
    ("Validation", val_idx),
    ("Final test", test_idx)
]:
    print(f"{name}: {len(idx):,}")

assert len(set(nuisance_idx) & set(test_idx)) == 0
assert len(set(cate_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

def make_forest(seed):
    return RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=80,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed
    )

# Step 1: nuisance outcome models.
mu = {}

for group in [0, 1]:
    idx = nuisance_idx[t[nuisance_idx] == group]
    model = make_forest(SEED + group)
    model.fit(X.iloc[idx], y[idx])
    mu[group] = model
    print(f"Nuisance model {group} trained", flush=True)

# Known randomized assignment probability:
# approximately 1/3 for each of the original three arms.
# Within the two-arm subset, P(T=1) = 1/2.
p = 0.5

# Step 2: doubly robust pseudo-outcome
# on a separate CATE-training sample.
x_cate = X.iloc[cate_idx]
t_cate = t[cate_idx]
y_cate = y[cate_idx]

m0 = mu[0].predict(x_cate)
m1 = mu[1].predict(x_cate)

pseudo = (
    m1 - m0
    + t_cate * (y_cate - m1) / p
    - (1 - t_cate) * (y_cate - m0) / (1 - p)
)

dr_model = make_forest(SEED + 10)
dr_model.fit(x_cate, pseudo)

print("DR-Learner trained", flush=True)

# Step 3: select targeting policy on validation.
def policy_matrix(predictions):
    n = len(predictions)
    order = np.argsort(-predictions, kind="stable")

    policies = {
        "None": np.zeros(n, dtype=int),
        "Everyone": np.ones(n, dtype=int)
    }

    for fraction in [0.2, 0.4, 0.6, 0.8]:
        selected = np.zeros(n, dtype=int)
        count = int(round(fraction * n))
        selected[order[:count]] = 1
        policies[f"Top {int(fraction*100)}%"] = selected

    return policies

def evaluate(idx, predictions, label):
    observed_t = t[idx]
    observed_y = y[idx]

    contrast = (
        observed_t * observed_y / p
        - (1 - observed_t) * observed_y / (1-p)
    )

    rows = []
    policies = policy_matrix(predictions)

    for name, action in policies.items():
        contributions = action * (contrast - COST)
        estimate = contributions.mean()
        se = contributions.std(ddof=1) / np.sqrt(len(idx))

        rows.append({
            "split": label,
            "policy": name,
            "target_fraction": action.mean(),
            "net_value": estimate,
            "ci95_low": estimate - 1.96 * se,
            "ci95_high": estimate + 1.96 * se
        })

    return pd.DataFrame(rows)

val_pred = dr_model.predict(X.iloc[val_idx])
validation = evaluate(
    val_idx, val_pred, "validation"
)

winner = validation.loc[
    validation["net_value"].idxmax(), "policy"
]

print("\n=== VALIDATION POLICY RESULTS ===")
print(validation.round(5).to_string(index=False))
print("\nSelected policy:", winner)

# Step 4: final evaluation, without refitting or
# changing the selected policy.
test_pred = dr_model.predict(X.iloc[test_idx])
test_results = evaluate(
    test_idx, test_pred, "final_test"
)

selected_test = test_results[
    test_results["policy"] == winner
]

everyone_test = test_results[
    test_results["policy"] == "Everyone"
]

print("\n=== UNTOUCHED FINAL TEST ===")
print("Selected policy:", winner)
print(selected_test.round(5).to_string(index=False))
print("\nEveryone baseline:")
print(everyone_test.round(5).to_string(index=False))

# Paired policy comparison on the final test.
actions = policy_matrix(test_pred)
a_selected = actions[winner]
a_everyone = actions["Everyone"]

tt = t[test_idx]
yy = y[test_idx]
contrast = tt * yy / p - (1-tt) * yy / (1-p)

paired = (
    (a_selected - a_everyone) * (contrast - COST)
)

diff = paired.mean()
se = paired.std(ddof=1) / np.sqrt(len(paired))

print("\n=== PAIRED FINAL COMPARISON ===")
print("Selected minus Everyone:", round(diff, 6))
print(
    "95% CI:",
    round(diff - 1.96*se, 6),
    round(diff + 1.96*se, 6)
)

validation.to_csv(
    OUTPUT / "hillstrom_dr_validation.csv",
    index=False
)

test_results.to_csv(
    OUTPUT / "hillstrom_dr_final_test.csv",
    index=False
)

pd.DataFrame({
    "predicted_cate": test_pred,
    "treatment": tt,
    "spend": yy,
    "selected_action": a_selected
}).to_parquet(
    OUTPUT / "hillstrom_dr_test_predictions.parquet",
    index=False
)

print("\nHILLSTROM DR-LEARNER COMPLETED")
