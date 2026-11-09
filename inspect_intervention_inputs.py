from pathlib import Path
import pandas as pd

root = Path("Hillstrom_Dataset/Processed")

files = [
    "hillstrom_dr_validation_predictions.parquet",
    "hillstrom_hurdle_validation_predictions.parquet",
    "hillstrom_dr_test_predictions.parquet",
    "hillstrom_policy_evaluation.csv",
    "hillstrom_paired_policy_comparisons.csv",
    "hillstrom_model_agreement.csv",
    "hillstrom_experimental_ate.csv",
]

for filename in files:
    path = root / filename

    print("\n" + "=" * 65)
    print(filename)
    print("=" * 65)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
