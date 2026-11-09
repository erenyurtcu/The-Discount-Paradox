from pathlib import Path
import numpy as np
import pandas as pd
from math import erfc, sqrt

root = Path(__file__).resolve().parent
raw_files = sorted((root / "Hillstrom_Dataset").glob("*MineThatData*.csv"))
assert len(raw_files) == 1, f"Expected one raw Hillstrom file, found {len(raw_files)}"
df = pd.read_csv(raw_files[0]); df.columns = df.columns.str.strip().str.lower()
assert len(df) == 64000
saved = pd.read_csv(root / "Hillstrom_Dataset/Processed/hillstrom_experimental_ate.csv")
checks = 0
for treatment in ("Mens E-Mail", "Womens E-Mail"):
    for outcome in ("visit", "conversion", "spend"):
        a = df.loc[df.segment == treatment, outcome].astype(float)
        b = df.loc[df.segment == "No E-Mail", outcome].astype(float)
        ate = a.mean()-b.mean()
        se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        row = saved.loc[(saved.treatment == treatment) & (saved.outcome == outcome)]
        assert len(row) == 1
        row = row.iloc[0]
        for col, val in (("ate",ate),("standard_error",se),("ci95_low",ate-1.96*se),("ci95_high",ate+1.96*se)):
            assert np.isclose(float(row[col]),val,rtol=1e-8,atol=1e-10), (treatment,outcome,col)
        checks += 1
print(f"PASS: {checks}/6 Hillstrom ATE outcomes, including SE and CI")
print("No source files or saved outputs modified.")
